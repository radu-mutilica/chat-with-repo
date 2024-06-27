import logging
import os
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from httpx import AsyncClient
from starlette.responses import StreamingResponse

from libs.models import RequestData
from libs.rag import answer_query
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


async def get_client():
    """Helper func to keep a client hot"""
    async with httpx.AsyncClient() as client:
        yield client


@app.post("/chat/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def chat_with_repo(request: RequestData, client: AsyncClient = Depends(get_client)):
    """Endpoint for chatting with your repo.

    Get the user's search string, build the context, format the prompt and issue the assistant call.

    Args:
        request: (RequestData) the request.
        client: (httpx.AsyncClient) the client.
    """
    try:
        rag_response = await answer_query(request.last_message(), request.history(), client)

        # Wait for the first response chunk. This helps with profiling when looking at charts.
        first_chunk = await rag_response.stream.__anext__()
        return StreamingResponse(
            async_chain(first_chunk, rag_response.stream),
            media_type="text/html"
        )

    except AssertionError:
        raise HTTPException(status_code=400, detail='Not a valid repository')

    except Exception:
        logger.exception(f'Failed to process request: {request}')
        raise HTTPException(status_code=500, detail='An error occurred while processing the query')


