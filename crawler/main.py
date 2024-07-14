import asyncio
import logging
import os
import tempfile

import db
from libs import splitting, crawl_targets
from libs.http import OptimizedAsyncClient
from libs.models import RepoCrawlTarget
from libs.proxies import perform_task, summaries
from libs.proxies.embeddings import HFEmbeddingFunc
from repository import load_repo, check_if_crawl_needed

log_level = os.environ['LOG_LEVEL']
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


async def crawl_repo(
        crawl_details: RepoCrawlTarget,
        client: OptimizedAsyncClient,
        emb_func: HFEmbeddingFunc,
):
    """Main crawler function.

    This holds most of the crawling logic, while using LLM calls to also summarize various
    entities (the repo itself, files and code snippets).

    Args:
        crawl_details: (RepoCrawlDetails): the details of the repo to crawl (url, branch, etc)
        client (OptimizedAsyncClient): The httpx client to use.
        emb_func (HFEmbeddingFunc): The embedding function to use for crawling.

    Once crawled and processed, insert everything into a chroma collection.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f'Loading repository "{crawl_details.url}:{crawl_details.branch}" at {tmp_dir}')
        repo = await load_repo(crawl_details.url, crawl_details.branch, tmp_dir)

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

        chunks = await splitting.split_documents(
            repo=repo,
            client=client
        )

        with db.VectorDBCollection(crawl_details.target_collection, emb_func) as vecdb_client:
            vecdb_client.add(
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[chunk.metadata['vecdb_idx'] for chunk in chunks],

            )


async def crawl(targets):
    """Helper function to create all crawling tasks (one per repo defined in the yaml file)"""
    client = OptimizedAsyncClient()
    emb_func = HFEmbeddingFunc(client)

    tasks = [
        asyncio.create_task(crawl_repo(
            crawl_details=repo_crawl_details,
            client=client,
            emb_func=emb_func
        ))
        async for repo_crawl_details in check_if_crawl_needed(targets, client)
    ]

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(crawl(crawl_targets))
