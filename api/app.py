import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from httpx import AsyncClient
from starlette.responses import StreamingResponse

import rag
from libs import crawl_targets
from libs.models import RequestData, Message
from libs.proxies import stream_task, perform_task, rephraser
from libs.proxies.chat import ChatWithRepo
from libs.utils import register_profiling_middleware

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

sim_search_top_k = int(os.environ['SIM_SEARCH_TOP_K'])
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
        return answer_query(request.last_message(), request.history(), client)

    except AssertionError:
        raise HTTPException(status_code=400, detail='Not a valid repository')

    except Exception:
        raise HTTPException(status_code=500, detail='An error occurred while processing the query')


async def answer_query(
        last_message: Message,
        chat_history: List[Message],
        client: AsyncClient) -> StreamingResponse:
    """Do some query processing first, check if there is any chat history, create
    an llm prompt based on the previous facts and send it over to the llm.

    Args:
        last_message (Message): the last message aka the user query.
        chat_history (List[Message]): the chat history, might be an empty list.
        client: (httpx.AsyncClient) the client.

    Returns:
        StreamingResponse from the LLM.
    """
    # todo: Assume all messages are about the same repo
    query, subnet = last_message.content.query, last_message.content.repo

    assert subnet in crawl_targets, 'Not a valid repo'

    # Check for a chat history, and if present, rephrase the query given the history.
    # This step is important to guarantee good simsearch results further down
    if chat_history:
        start = time.time()
        logger.info('Found a chat history, rephrasing last query...')
        rag_query = await perform_task(
            rephraser.RephraseGivenHistory(
                query=query,
                chat_history=chat_history),
            client=client)
        logger.info(f'Rephrasing task took {time.time() - start:.2f} seconds!')
    else:
        rag_query = query

    start = time.time()
    context = await rag.context_pipeline(
        collection=crawl_targets[subnet]['target_collection'],
        query=rag_query,
        sim_top_k=sim_search_top_k,
        client=client
    )
    logger.info(f"{time.time() - start:.2f}s: Context length={len(context)}:\n{context}")

    chat_with_repo_task = ChatWithRepo(
        question=rag_query,
        context=context,
        github_name=crawl_targets[subnet]['name'],
        repo_name=subnet

    )
    # Hardcode this to a streaming response. Once this model has support
    # for standard responses, we can fix this
    return StreamingResponse(stream_task(chat_with_repo_task))


