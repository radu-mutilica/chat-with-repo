import asyncio
import logging
import os
from typing import List

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, Language

from libs import extensions
from libs.http import OptimizedAsyncClient
from libs.models import Repo
from libs.proxies import summaries, perform_task

logger = logging.getLogger(__name__)

splitters = {}
contextual_window_snippet_radius = 2
vecdb_idx_fmt = "{source}:{index}"
extra_readme_append_fmt = """

***
Document path: {file_path}
Document contents:
{page_content}
***

"""


class MissingRootReadme(Exception):
    pass


class MultipleRootReadmes(Exception):
    pass


def prepare_splitter(language: Language) -> TextSplitter:
    """Helper function to keep a working list of text splitters."""
    global splitters

    if language not in splitters:
        splitters[language] = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=512,
            chunk_overlap=0
        )
    return splitters[language]


async def split_document(
        document: Document,
        repo: Repo,
        client: OptimizedAsyncClient) -> List[Document]:
    """Most of the heavy lifting associated with splitting and summarizing files and code snippets.
    
    This async func does the following processing steps:
    
    - Find the file's extension (to prepare the correct text splitter)
    - Summarize the file.
    - Split the file into code chunks.
    - Summarize each code chunk separately, using context like the aforementioned file summary.
    - Return the chunks as well as the file summary (merged into a list of Documents).
    
    Note:
        For increased accuracy in final results, we ended up taking the approach of hiding away
        the raw code in the document's metadata, instead using the summary as the page_content,
        which is passed to the embedding function.
        
        When we build the final context for the prompt, we make sure to insert both snippet
        summaries and raw code.
    
    Args:
        document: (Document) the document to split.
        repo: (Repo) the repo containing the document.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A list of documents (chunks).
    """
    extension = os.path.splitext(document.metadata["source"])[1]
    language = extensions.identify_language(extension)

    file_summary = perform_task(
        summaries.SummarizeFile(
            repo_name=repo.name,
            repo_summary=repo.summary,
            tree=repo.tree,
            file_path=document.metadata['file_path'],
            content=document.page_content,
            language=language
        ),
        client=client
    )
    splitter = prepare_splitter(language=language)
    snippets = splitter.create_documents([document.page_content])

    # Swap the content for just the summary
    document.metadata['original_page_content'] = document.page_content
    try:
        document.page_content = await file_summary
    except httpx.HTTPError as e:
        logger.error(f"Failed to summarize document {document.metadata['file_path']}: {str(e)}")
        document.page_content = ''

    document.metadata['language'] = language
    document.metadata['document_type'] = 'file-summary'
    document.metadata['vecdb_idx'] = vecdb_idx_fmt.format(
        source=document.metadata['file_path'],
        index='summary'
    )

    for idx, snippet in enumerate(snippets):
        snippet.metadata.update(document.metadata)
        snippet_summary = perform_task(
            summaries.SummarizeSnippet(
                repo_name=repo.name,
                repo_summary=repo.summary,
                tree=repo.tree,
                language=language,
                file_path=snippet.metadata['file_path'],
                file_summary=document.page_content,
                context=wrap_code_snippet_with_neighbours(idx, snippets),
                content=snippet.page_content),
            client=client
        )
        # Store the "raw code" in the metadata, use it when building context
        # but use the summary for sim search
        snippet.metadata['original_page_content'] = snippet.page_content
        snippet.metadata['document_type'] = 'code-snippet'
        snippet.metadata['vecdb_idx'] = vecdb_idx_fmt.format(
            source=document.metadata['file_path'],
            index=idx
        )
        snippet.page_content = await snippet_summary

    # Only add document if we have a summary for it
    if document.page_content:
        snippets.insert(0, document)

    return snippets


async def split_documents(
        repo: Repo,
        client: OptimizedAsyncClient
) -> List[Document]:
    """Wrapper function for building a list of coroutines and executing them"""
    chunks = []

    tasks = [
        split_document(document, repo, client) for document in repo.documents
    ]

    for task in asyncio.as_completed(tasks):
        chunks.extend(await task)

    return chunks


def wrap_code_snippet_with_neighbours(snippet_index: int, code_snippets: List[Document]) -> str:
    """Given a list of code snippets Documents and the index of one of them, return a string
    representing the indexed snippet wrapped (prepended and appended) with its immediate
    neighbours. For example if we have a list [1, 2, 3, 4, 5] and the snippet index is 4, we return
    the concatenation resulting from snippets 2, 3, 4, and 5. This is used in the summary step
    for more context about the snippet.

    Args:
        snippet_index: (int) the index of the snippet to wrap.
        code_snippets: (list) the list of code snippets in the file.

    Returns:
        str: the wrapped snippet.
    """
    min_idx = snippet_index - contextual_window_snippet_radius
    min_idx = max(0, min_idx)

    max_idx = snippet_index + contextual_window_snippet_radius

    return '\n'.join((doc.page_content for doc in code_snippets[min_idx:max_idx]))


def merge_readmes(main_readme: str, other_md_files: List[Document]) -> str:
    """Given a main Readme.md document, and a list of other readmes found in the repo, check
    if the main readme file links to or mentions the secondary ones, and if so, insert their
    contents into the Document.page_content of the main_readme.

    Essentially we are merging together multiple page_contents.

    Args:
        main_readme: (Document) the main readme.md document.
        other_md_files: (List[Document]) a list of other readmes.

    Returns:
        str: the main readme document merged with other referenced markdown files.
    """
    references = {}

    for line in main_readme.splitlines():
        for md_file in other_md_files:
            if md_file.metadata['file_path'] in line:
                references[md_file.metadata['file_path']] = md_file
                break

    if references:
        for file_path, md_file in references.items():
            main_readme += extra_readme_append_fmt.format(
                file_path=file_path,
                page_content=md_file.page_content
            )

    return main_readme


def expand_root_readme(documents: List[Document]) -> str:
    """Traverse the list of documents and try to find all the markdown files as well as the root
    readme, merging them into a big readme file.

    Args:
        documents: (list) A list of documents, some of which might be markdown files.

    Returns:
        The merged root readme document with other referenced markdown files.

    Raises:
        MultipleRootReadmes: if multiple root readmes are found.
        MissingRootReadme: if the root readme is not found.
    """
    extra_md_files = []
    root_readme = None

    for index, document in enumerate(documents):
        if document.metadata['file_path'].lower() == 'readme.md':
            # This must be the root repo readme file
            if root_readme is None:
                root_readme = document.page_content
            else:
                raise MultipleRootReadmes('Found multiple root readmes, can be only one')

        elif document.metadata['file_name'].lower().endswith('.md'):
            # These are potentially other readme files, will check below if referenced
            extra_md_files.append(document)

    if root_readme:
        if extra_md_files:  # Insert their contents into the root readme
            root_readme = merge_readmes(root_readme, extra_md_files)
    else:
        raise MissingRootReadme('no root readme.md found, unable to summarize repo')

    return root_readme
