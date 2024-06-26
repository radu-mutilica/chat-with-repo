import os
import logging
from datetime import datetime
from typing import AsyncGenerator
from urllib.parse import urlparse

import httpx
from directory_tree import display_tree
from langchain_community.document_loaders import GitLoader

import db
from libs.models import Repo
from libs.storage import vectordb

logger = logging.getLogger(__name__)

github_api_url = 'https://api.github.com/repos/'
mongo_url = f'mongodb://{os.environ["MONGO_HOST"]}:{os.environ["MONGO_PORT"]}/stats'
github_access_token = os.environ['GITHUB_API_KEY']
github_api_headers = {"Authorization": f"token {github_access_token}"}
timeout = httpx.Timeout(20.0, read=None)


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


async def get_last_commit_ts(url: str, branch: str, client: httpx.AsyncClient) -> int:
    """Helper function to get the last commit timestamp for a repo"""
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
    all_collections = [c.name for c in vectordb.list_collections()]
    logger.info(f'All collections: {all_collections}')
    stats = db.StatsDB(mongo_url)

    for subnet, crawl_config in targets.items():

        will_crawl = False
        last_commit_ts = await get_last_commit_ts(
            crawl_config['url'], crawl_config['branch'], client)

        # If no collection present, then just go ahead and crawl
        target_collection = crawl_config['target_collection']
        if target_collection not in all_collections:
            logger.info(f'Collection missing target={target_collection}. Crawling...')
            will_crawl = True
        else:  # If collection exists, check the latest commit timestamp
            logger.info(f'Collection present target={target_collection}. Checking last commit...')

            last_crawled_commit_ts = stats.get_last_commit(subnet)
            if last_crawled_commit_ts < last_commit_ts:
                logger.info(f'Old last commit ts={last_crawled_commit_ts} target={subnet}, '
                            f'new one {last_commit_ts}. Crawling...')
                will_crawl = True
            else:
                logger.info(f'Skipping target={target_collection}. Last commit @ {last_commit_ts}')

        if will_crawl:
            yield subnet, crawl_config
            stats.set_last_commit(subnet, last_commit_ts)
