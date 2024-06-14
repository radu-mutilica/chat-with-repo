import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import AsyncGenerator
from urllib.parse import urlparse

import httpx
from directory_tree import display_tree
from langchain_community.document_loaders import GitLoader

import db
from libs import splitting, crawl_targets
from libs.models import Repo
from libs.proxies import perform_task, summaries
from libs.storage import vectordb

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
github_api_url = 'https://api.github.com/repos/'
mongo_url = f'mongodb://{os.environ["MONGO_HOST"]}:{os.environ["MONGO_PORT"]}/stats'
github_access_token = os.environ['GITHUB_API_KEY']
github_api_headers = {
    "Authorization": f"token {github_access_token}"
}

timeout = httpx.Timeout(20.0, read=None)


async def crawl_repo(url: str, branch: str, target_collection: str, client: httpx.AsyncClient):
    """Main crawler function.

    This holds most of the crawling logic, while using LLM calls to also summarize various
    entities (the repo itself, files and code snippets).

    Args:
        url (str): The repository URL.
        branch (str): The repository branch.
        target_collection (str): The target collection for the database.
        client (httpx.AsyncClient): The httpx client to use.

    Once crawled and processed, insert everything into a chroma collection.
    """
    with tempfile.TemporaryDirectory() as local_path:
        logger.info(f'Loading repository "{url}:{branch}" at {local_path}')
        repo = await load_repo(url, branch, local_path)

        # Find and expand the root readme file to embed all the other referenced .md files.
        # This block of (hopefully) high-level repo knowledge is used to perform a repo summary.
        expanded_readme = splitting.expand_root_readme(repo.documents)
        repo_summary_task = summaries.SummarizeRepo(
            content=expanded_readme,
            repo_name=repo.name,
            tree=repo.tree,
        )
        repo_summary = await perform_task(repo_summary_task, client)
        repo.summary = repo_summary
        logger.info(f'Summarized repo summary task: {repo_summary}')

        chunks = await splitting.split_documents(
            repo=repo,
            client=client
        )

        logger.info(f'Found {len(chunks)} chunks')
        with db.VectorDBCollection(collection_name=target_collection) as vecdb_client:
            vecdb_client.add(
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[chunk.metadata['vecdb_idx'] for chunk in chunks],

            )


async def get_default_branch(repo_url: str, client: httpx.AsyncClient) -> str:
    """Helper function to figure out the default branch for a repo"""
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')

    if len(path_parts) != 2:
        raise ValueError(f'Invalid GitHub repository URL "{repo_url}"')

    owner, repo = path_parts

    url = github_api_url + f'{owner}/{repo}'

    response = await client.get(url, headers=github_api_headers, timeout=timeout)

    response.raise_for_status()
    repo_info = response.json()
    default_branch = repo_info['default_branch']
    logger.info(f'Repo: {repo_url} found default branch: {default_branch}')

    return default_branch


async def load_repo(url: str, branch, temp_path: str) -> Repo:
    """Helper function to load a git repo"""
    name = url.rsplit('/', maxsplit=1)[-1]
    root_path = f'{temp_path}/{name}'
    os.makedirs(root_path)

    git_loader = GitLoader(clone_url=url, repo_path=root_path, branch=branch)

    repo = Repo(
        name=name,
        branch=branch,
        url=url,
        documents=git_loader.load(),
        tree=display_tree(root_path, string_rep=True),
    )
    return repo


async def crawl(targets):
    """Helper function to create all crawling tasks (one per repo defined in the yaml file)"""
    client = httpx.AsyncClient()

    tasks = [
        asyncio.create_task(crawl_repo(
            url=crawl_config['url'],
            branch=crawl_config['branch'],
            target_collection=crawl_config['target_collection'],
            client=client
        ))
        async for subnet, crawl_config in check_if_crawl_needed(targets, client)
    ]

    await asyncio.gather(*tasks)


async def get_last_commit_ts(url: str, branch: str, client: httpx.AsyncClient) -> int:
    repo_parts = url.rstrip('/').split('/')
    owner = repo_parts[-2]
    repo = repo_parts[-1]

    url = github_api_url + f'{owner}/{repo}/commits/{branch}'

    response = await client.get(url, headers=github_api_headers, timeout=timeout)
    response.raise_for_status()

    last_commit = response.json()['commit']['author']['date']
    last_commit = datetime.strptime(last_commit, "%Y-%m-%dT%H:%M:%SZ")

    return int(last_commit.timestamp())


async def check_if_crawl_needed(targets, client: httpx.AsyncClient) -> AsyncGenerator:
    all_collections = vectordb.list_collections()
    stats = db.StatsDB(mongo_url)

    for subnet, crawl_config in targets.items():

        will_crawl = False
        last_commit_ts = await get_last_commit_ts(
            crawl_config['url'], crawl_config['branch'], client)

        # If no collection present, then just go ahead and crawl
        if not crawl_config['target_collection'] in all_collections:
            logger.info(f'Collection missing target={subnet}. Crawling...')
            will_crawl = True
        else:  # If collection exists, check the latest commit timestamp
            logger.info(f'Collection present target={subnet}. Checking last commit...')

            last_crawled_commit_ts = stats.get_last_commit(subnet)
            if last_crawled_commit_ts < last_commit_ts:
                logger.info(f'Old last commit ts={last_crawled_commit_ts} target={subnet}, '
                            f'new one {last_commit_ts}. Crawling...')
                will_crawl = True
            else:
                logger.info(f'Skipping target={subnet}. Latest commit @ {last_commit_ts}')

        if will_crawl:
            yield subnet, crawl_config
            stats.set_last_commit(subnet, last_commit_ts)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(crawl(crawl_targets))
