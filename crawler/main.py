import asyncio
import logging
import os
import tempfile

import httpx
from directory_structure import Tree
from langchain_community.document_loaders import GitLoader

from libs import splitting
from libs import storage
from libs.models import Repo
from libs.proxies import summaries

logger = logging.getLogger('crawler')

github_url = os.environ['GITHUB_URL']

enriched_page_content_fmt = '#{file_path}\n#{summary}\n{content}'
extra_readme_append_fmt = """
***
Document path: {file_path}
Document contents:
{page_content}
***
"""


def collection_exists():
    db = storage.vector_db
    return db._collection.count() > 1


def load_repo(url: str, temp_path: str, branch='main') -> Repo:
    git_loader = GitLoader(clone_url=url, repo_path=temp_path, branch=branch)
    repo = Repo(
        name=url.rsplit('/', maxsplit=1)[-1],
        branch=branch,
        url=url,
        documents=git_loader.load(),
        tree=str(Tree(temp_path, absolute=False))
    )
    return repo


async def main():
    client = httpx.AsyncClient()
    with tempfile.TemporaryDirectory() as local_path:
        repo = load_repo(github_url, local_path, branch="main")
        try:
            readme = splitting.find_readme(repo.documents)
        except splitting.MissingReadme:
            # todo: what do if repo is missing a main readme file?
            raise
        else:
            logger.debug(f'Found root readme file {readme.metadata["file_path"]}')
            repo_summary_task = summaries.SummarizeRepo(
                content=readme.metadata['enriched_page_content'],
                repo_name=repo.name,
                tree=repo.tree,
            )
            repo_summary = await summaries.produce(repo_summary_task, client)
            logger.debug(f'Summarized repo summary task: {repo_summary}')

            repo.metadata['summary'] = repo_summary
            chunks = await splitting.split_documents(
                documents=repo.documents,
                repo=repo,
                client=client
            )

            db = storage.vector_db
            logger.debug(f'Found {len(chunks)} chunks')
            db.add_documents(chunks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
