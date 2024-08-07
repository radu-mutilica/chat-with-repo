import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List
from urllib.parse import urlparse

import httpx
from directory_tree import display_tree
from langchain_community.document_loaders import GitLoader

import libs.stats
from libs.models import Repo, RepoCrawlStats, RepoCrawlTarget
from libs.storage import vector_db

logger = logging.getLogger(__name__)

# todo: clean this junk
github_api_url = 'https://api.github.com/repos/'
github_access_token = os.environ['GITHUB_API_KEY']
github_api_headers = {"Authorization": f"token {github_access_token}"}
timeout = httpx.Timeout(20.0, read=None)


async def get_default_branch(repo_url: str, client: httpx.AsyncClient) -> str:
    """Helper function to figure out the default branch for a repo, using the GitHub API"""
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


async def get_repo_metadata(
        crawl: RepoCrawlTarget,
        client: httpx.AsyncClient
) -> RepoCrawlStats:
    """Query GitHub API to get detailed information about a specific repo/branch combo.

    Will do two separate requests, one for the general repo, one for the specific branch.

    Args:
        crawl (RepoCrawlTarget): the details of the repo/branch combo to crawl (like url, branch)
        client: An instance of the `httpx.AsyncClient` class used for making asynchronous
        HTTP requests.

    Returns:
        An instance of the `RepoMetadata` model containing the metadata of the repository.

    """
    repo_parts = crawl.url.rstrip('/').split('/')

    owner = repo_parts[-2]
    repo = repo_parts[-1]

    data = await _get_repo_information(owner, repo, client)
    last_commit_ts = await get_last_commit_ts(owner, repo, crawl.branch, client)

    data['branch'] = {
        'name': crawl.branch,
        'last_commit_ts': last_commit_ts,
    }
    data['github_id'] = data.pop('id')
    data['tag'] = crawl.tag
    data['repo_id'] = crawl.repo_id

    metadata = RepoCrawlStats.model_validate(data)

    return metadata


async def _get_repo_information(owner: str, repo: str, client: httpx.AsyncClient) -> dict:
    """Helper func to call the GitHub API and get general repo details."""
    url = github_api_url + f'{owner}/{repo}'

    response = await client.get(url, headers=github_api_headers, timeout=timeout)
    response.raise_for_status()

    return response.json()


async def get_last_commit_ts(
        owner: str,
        repo: str,
        branch: str,
        client: httpx.AsyncClient) -> float:
    """Get the last commit timestamp for a specific repo/branch combo.

    Args:
        owner: The username or organization name of the repository owner.
        repo: The name of the repository.
        branch: The name of the branch.
        client: The httpx.AsyncClient object used for sending HTTP requests.

    Returns:
        float: The timestamp of the last commit on the specified branch in the repository.

    Raises:
        httpx.HTTPStatusError: If the HTTP request to the GitHub API fails or returns an
        error status.
    """
    url = github_api_url + f'{owner}/{repo}/commits/{branch}'

    response = await client.get(url, headers=github_api_headers, timeout=timeout)
    response.raise_for_status()

    last_commit = response.json()['commit']['author']['date']
    last_commit = datetime.strptime(last_commit, "%Y-%m-%dT%H:%M:%SZ")

    return last_commit.timestamp()


async def check_if_crawl_needed(
        crawl_targets: List[RepoCrawlTarget],
        client: httpx.AsyncClient
) -> AsyncGenerator:
    """We don't want to crawl every repo on every cronjob, so check for new commits first.

    This method checks if a crawl is needed for each target in the provided dictionary. It
    determines whether a crawl is needed based on the presence of target collections and the latest
    commit timestamps. If a crawl is needed, it yields the subnet and crawl configuration.

    Args:
        crawl_targets: (List[RepoCrawlTarget]) A list with the targets to be crawled.
        client: An httpx.AsyncClient instance used for making asynchronous HTTP requests.

    Returns:
        An asynchronous generator that yields tuples of subnet and crawl configuration if crawl is
        needed.
    """
    stats = libs.stats.CrawlStats()

    all_collections = [c.name for c in vector_db.list_collections()]
    logger.info(f'All collections: {all_collections}')

    for crawl in crawl_targets:
        will_crawl = False  # assume we don't need to crawl

        fresh_metadata = await get_repo_metadata(crawl, client)

        if crawl.target_collection not in all_collections:
            # If no collection present, then just go ahead and crawl
            logger.info(f'Collection missing target={crawl.target_collection}. Crawling...')
            will_crawl = True

        else:
            # If collection exists, check the latest commit timestamp
            logger.info(f'Collection found, target={crawl.target_collection}. Checking last commit')

            try:
                last_crawl_run = stats.get_repo_stats(crawl.repo_id)
                if last_crawl_run.branch.last_commit_ts < fresh_metadata.branch.last_commit_ts:
                    logger.info(
                        f'Stale last commit ts={last_crawl_run.branch} target={crawl.repo_id}, '
                        f'new one {fresh_metadata.branch}. Crawling...')
                    will_crawl = True
                else:
                    logger.info(f'Skipping target={crawl.target_collection}. Last commit @ '
                                f'{fresh_metadata.branch}')
            except libs.stats.NoStatsFound:
                logger.error('Missing stats for target={crawl.target_collection}. Crawling...')
                will_crawl = True

        if will_crawl or os.getenv('FORCE_CRAWL') == 'TRUE':
            try:
                yield crawl
            except Exception:
                logger.exception(f'Failed to crawl target={crawl}:')
            else:
                stats.update_crawl_stats(crawl.repo_id, fresh_metadata)
