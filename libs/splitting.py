import asyncio
import logging
import os
from typing import List

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, Language

from libs import extensions
from libs.models import Repo
from libs.proxies import summaries, perform

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
    global splitters

    if language not in splitters:
        splitters[language] = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1024,
            chunk_overlap=0
        )
    return splitters[language]


async def split_document(document, repo, client) -> List[Document]:
    extension = os.path.splitext(document.metadata["source"])[1]
    language = extensions.identify_language(extension)

    file_summary = perform(
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
    except httpx.HTTPError:
        # Failed to summarize file, most likely because it exceeds token limit
        # todo: better error handling, diagnostics here
        logger.error(f"Failed to summarize document {document.metadata['file_path']}")
        document.page_content = ''

    document.metadata['language'] = language
    document.metadata['vecdb_idx'] = vecdb_idx_fmt.format(
        source=document.metadata['file_path'],
        index='main'
    )

    for idx, snippet in enumerate(snippets):
        snippet.metadata.update(document.metadata)
        snippet_summary = perform(
            summaries.SummarizeSnippet(
                repo_name=repo.name,
                repo_summary=repo.metadata['summary'],
                tree=repo.tree,
                language=language,
                file_path=snippet.metadata['file_path'],
                file_summary=document.page_content,
                context=build_context(idx, snippets),
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
    snippets = []

    tasks = [
        split_document(document, repo, client) for document in documents
    ]

    for task in asyncio.as_completed(tasks):
        results = await task
        snippets.extend(results)

    return snippets


def build_context(current_index, document_snippets):
    min_idx = current_index - contextual_window_snippet_radius
    min_idx = max(0, min_idx)

    max_idx = current_index + contextual_window_snippet_radius

    return '\n'.join((doc.page_content for doc in document_snippets[min_idx:max_idx]))


def merge_readmes(readme, other_readmes):
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

    readme.metadata['enriched_page_content'] = enriched_page_content

    return readme


def find_readme(documents: List[Document]) -> Document:
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
