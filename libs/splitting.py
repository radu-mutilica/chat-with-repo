import asyncio
import logging
import os
from typing import List

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, Language

from libs import extensions
from libs.models import Repo
from libs.proxies import summaries, perform_task

logger = logging.getLogger(__name__)

splitters = {}
contextual_window_snippet_radius = 2

summary = """
# {file_path}
# {file_summary}
# {snippet_summary}

{content}
"""
extra_readme_append_fmt = """
***
Document path: {file_path}
Document contents:
{page_content}
***
"""

vecdb_idx_fmt = "{source}:{index}"


def prepare_splitter(language: Language) -> TextSplitter:
    """Helper function to keep a working list of text splitters."""
    global splitters

    if language not in splitters:
        splitters[language] = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1024,
            chunk_overlap=0
        )
    return splitters[language]


async def split_document(
        document: Document,
        repo: Repo, client:
        httpx.AsyncClient) -> List[Document]:
    """Most of the heavy lifting associated with splitting and summarizing files and code snippets.
    
    This async func does the following processing steps:
    
    - Find the file's extension (to prepare the correct text splitter)
    - Summarize the file.
    - Split the file into code chunks.
    - Summarize each code chunk separately, using context like the aforementioned file summary.
    - Return the chunks as well as the file.
    
    Note:
        For increased accuracy in final results, we ended up taking the approach of hiding away
        the raw code in the document's metadata, instead using the summary as the page_content,
        which is passed to the embedding function.
        
        When we build the final context for the prompt, we make sure to insert both snippet
        summaries as well as raw code.
    
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
            repo_summary=repo.metadata['summary'],
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
        # Failed to summarize file, most likely because it exceeds token limit
        # todo: better error handling, diagnostics here
        logger.error(f"Failed to summarize document {document.metadata['file_path']}: {str(e)}")
        document.page_content = ''

    document.metadata['language'] = language
    document.metadata['vecdb_idx'] = vecdb_idx_fmt.format(
        source=document.metadata['file_path'],
        index='main'
    )

    for idx, snippet in enumerate(snippets):
        snippet.metadata.update(document.metadata)
        snippet_summary = perform_task(
            summaries.SummarizeSnippet(
                repo_name=repo.name,
                repo_summary=repo.metadata['summary'],
                tree=repo.tree,
                language=language,
                file_path=snippet.metadata['file_path'],
                file_summary=document.page_content,
                context=wrap_code_snippet_with_neighbours(idx, snippets),
                content=snippet.page_content),
            client=client
        )
        # Store the "code" in the metadata
        snippet.metadata['original_page_content'] = snippet.page_content
        snippet.metadata['vecdb_idx'] = vecdb_idx_fmt.format(
            source=document.metadata['file_path'],
            index=idx
        )
        snippet.page_content = await snippet_summary

    # No reason to add the document if we didn't compute a summary for it
    if document.page_content:
        snippets.insert(0, document)

    return snippets


async def split_documents(
        documents: List[Document],
        repo: Repo,
        client: httpx.AsyncClient
) -> List[Document]:
    """Wrapper function for async work"""
    chunks = []

    tasks = [
        split_document(document, repo, client) for document in documents
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


def merge_readmes(readme: Document, other_readmes: List[Document]) -> Document:
    """Given a main Readme.md document, and a list of other readmes found in the repo, check
    if the main readme file links to or mentions the secondary ones, and if so, insert their
    contents into the Document.page_content of the main_readme.

    Essentially we are merging together multiple page_contents.

    Args:
        readme: (Document) the main Readme.md document.
        other_readmes: (list of Document) a list of other readmes.

    Returns:
        Document: the merged document.
    """
    enriched_page_content = readme.page_content

    referenced_files = {}
    for line in readme.page_content.splitlines():
        for other_readme in other_readmes:
            if other_readme.metadata['file_path'] in line:
                referenced_files[other_readme.metadata['file_path']] = other_readme

    if referenced_files:
        for file_path, other_readme in referenced_files.items():
            enriched_page_content += extra_readme_append_fmt.format(
                file_path=file_path,
                page_content=other_readme.page_content
            )

    # We store the merged version in the metadata
    readme.metadata['enriched_page_content'] = enriched_page_content

    return readme


def find_readme(documents: List[Document]) -> Document:
    """Traverse the list of documents and try to find the main Readme.md file. If the Readme.md file
    mentions or links to other readme files, be sure to insert them into the page_contents too.

    Args:
        documents: (list) A list of documents, some of which might be Readme.md files.

    Returns:
        Document: The main Readme.md file.

    Raises:
        MissingReadme: If no main Readme.md file is found.
    """
    other_readmes = []
    root_readme = None

    for index, document in enumerate(documents):

        if document.metadata['file_path'].lower() == 'readme.md':
            # This must be the root repo readme file
            root_readme = document
        elif document.metadata['file_name'].lower().endswith('.md'):
            # These are some other readme files, probably relevant still
            other_readmes.append(document)

    if root_readme:
        if other_readmes:
            root_readme = merge_readmes(root_readme, other_readmes)

    else:
        raise MissingReadme('no main readme file found')

    return root_readme


class MissingReadme(Exception):
    pass
