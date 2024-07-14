import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from starlette.responses import StreamingResponse

from libs.http import OptimizedAsyncClient
from libs.models import RequestData, RepoCrawlStats
from libs.rag import answer_query
from libs.stats import CrawlStats
from libs.utils import register_profiling_middleware, async_chain

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

redis_url = os.environ['REDIS_URL']


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Simple rate limiting layer"""
    redis_connection = redis.from_url(redis_url, encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()


app = FastAPI(lifespan=lifespan)
register_profiling_middleware(app)


async def get_http_client():
    """Helper func to keep a client hot"""
    async with OptimizedAsyncClient() as client:
        yield client
        
        
async def get_stats_client():
    """Helper func to keep a client hot"""
    yield CrawlStats()


@app.get("/repos/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def repos(
        crawl_stats: CrawlStats = Depends(get_stats_client)
) -> List[RepoCrawlStats]:
    """Fetch a list of crawled repos from the crawl stats database.

    Args:
        crawl_stats: The CrawlStats object used to get repository information.

    Returns:
        List[RepoCrawlStats]: A list of RepoMetadata objects representing the retrieved repositories.
    """
    start = time.perf_counter()

    all_repos = crawl_stats.get_repos()

    time_elapsed = time.perf_counter() - start
    logger.info(f'Retrieved repos in {time_elapsed:.2f} seconds')

    return all_repos


@app.post("/chat/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def chat_with_repo(
        request: RequestData,
        client: OptimizedAsyncClient = Depends(get_http_client)
) -> StreamingResponse:
    """Endpoint for chatting with your repo.

    Get the user's search string, build the context, format the prompt and issue the assistant call.

    Args:
        request: (RequestData) the request.
        client: (OptimizedAsyncClient) the client to use throughout the pipeline.
    """
    try:
        start = time.perf_counter()

        rag_payload = await answer_query(request.last_message, request.history, client)
        rag_time_elapsed = time.perf_counter() - start
        logger.info(f'Got context in {rag_time_elapsed:.2f}s, {len(rag_payload.formatted)} tokens')

        # Wait for the first response chunk. This helps with profiling, exposing real runtime.
        first_chunk_start = time.perf_counter()
        first_chunk = await rag_payload.stream.__anext__()
        logger.info(f'Got first response chunk in {time.perf_counter() - first_chunk_start:.2f}s')

        logger.info(f'Total query time: {time.perf_counter() - start:.2f}s')
        return StreamingResponse(
            async_chain(first_chunk, rag_payload.stream),
            media_type="text/html"
        )

    except AssertionError:
        raise HTTPException(status_code=400, detail='Not a valid repository')

    except Exception:
        logger.exception(f'Failed to process request: {request}')
        raise HTTPException(status_code=500, detail='An error occurred while processing the query')
