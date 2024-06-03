import asyncio
import logging
import os
import tempfile

import httpx
from directory_structure import Tree
from langchain_community.document_loaders import GitLoader

import db
from libs import splitting
from libs.models import Repo
from libs.proxies import perform, summaries

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

github_url = os.environ['GITHUB_URL']


async def main():
    """Main crawler function.

    This holds most of the crawling logic, while using LLM calls to also summarize various
    entities (the repo itself, files and code snippets).

    Once crawled and processed, insert everything into a chroma collection.
    """
    client = httpx.AsyncClient()
    with tempfile.TemporaryDirectory() as local_path:
        logger.info(f'Loading following repository: {github_url} at {local_path}')
        repo = load_repo(github_url, local_path, branch="main")
        try:
            readme = splitting.find_readme(repo.documents)
        except splitting.MissingReadme:
            # todo: what do if repo is missing a main readme file? the whole thing revolves
            # around building context based on the initial repo summary, which might be impossible
            # without a solid repo readme file
            raise

        logger.info(f'Found root readme file {readme.metadata["file_path"]}')
        repo_summary_task = summaries.SummarizeRepo(
            content=readme.metadata['enriched_page_content'],
            repo_name=repo.name,
            tree=repo.tree,
        )
        repo_summary = await perform(repo_summary_task, client)
        logger.info(f'Summarized repo summary task: {repo_summary}')

        repo.metadata['summary'] = repo_summary
        chunks = await splitting.split_documents(
            documents=repo.documents,
            repo=repo,
            client=client
        )
        logger.info(f'Found {len(chunks)} chunks')
        with db.VectorDBCollection(
                collection_name=os.environ['CHROMA_COLLECTION']) as vecdb_client:
            vecdb_client.add(
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[chunk.metadata['vecdb_idx'] for chunk in chunks],

            )


def load_repo(url: str, temp_path: str, branch='main') -> Repo:
    """Helper function to load a git repo"""
    git_loader = GitLoader(clone_url=url, repo_path=temp_path, branch=branch)
    documents = git_loader.load()
    repo = Repo(
        name=url.rsplit('/', maxsplit=1)[-1],
        branch=branch,
        url=url,
        documents=documents,
        tree=str(Tree(temp_path, absolute=False))
    )
    return repo


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
