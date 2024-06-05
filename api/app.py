import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from httpx import AsyncClient
from langchain_core.documents import Document
from starlette.responses import StreamingResponse

from libs import crawl_targets
from libs import storage
from libs.models import RequestData
from libs.proxies import embeddings, reranker, stream_task
from libs.proxies.chat import format_context, ChatWithRepo

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

sim_search_top_k = int(os.environ['SIM_SEARCH_TOP_K'])
redis_url = os.environ['REDIS_URL']

app = FastAPI()


async def get_client():
    """Helper func to keep a client hot"""
    async with httpx.AsyncClient() as client:
        yield client


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Simple rate limiting layer"""
    redis_connection = redis.from_url(redis_url, encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()


@app.post("/chat/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def chat_with_repo(request: RequestData, client: AsyncClient = Depends(get_client)):
    """Endpoint for chatting with your repo.

    Get the user's search string, build the context, format the prompt and issue the LLM call.

    Args:
        request: (RequestData) the request.
        client: (httpx.AsyncClient) the client.
    """
    try:
        data = request.messages[0].content
        query, repo = data.query, data.repo

        assert repo in crawl_targets, "Not a valid repo"

        # Commented this out since the model used for answering the question is
        # pretty good at detecting 'erroneous' user input
        #
        # query_inspector_task = QueryInspector(
        #     user_input=query
        # )
        # query_inspector_response = await perform_task(task=query_inspector_task, client=client)
        # assert query_inspector_response.strip() == 'TRUE', 'Invalid user query'

        context = await build_rag_context(
            repo=repo,
            search_query=query,
            sim_top_k=sim_search_top_k,
            client=client
        )

        chat_with_repo_task = ChatWithRepo(
            question=query,
            context=context
        )

        logger.debug(f"Context generated length={len(context)}:\n{context}")
        # Hardcode this to a streaming response. Once corcel api has support
        # for standard responses, we can fix this
        return StreamingResponse(stream_task(chat_with_repo_task))

    except AssertionError:
        raise HTTPException(status_code=400, detail="Not a valid repository")

    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")


async def build_rag_context(
        repo: str,
        search_query: str,
        sim_top_k: int,
        client: httpx.AsyncClient) -> str:
    """Main logic for the RAG component.

    Use similarity search to gather a list of documents, then rerank them and format them into
    a prompt template.

    Args:
        repo: (str) the name of the repo to build the context around
        search_query: (str) the user's search query.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A string representing the context for the final llm prompt.
    """
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(search_query, client)
    logger.info(f'Got embedding for user search query, took {round(time.time() - start, 2)}s')

    logger.info(f'Searching collection {repo}')
    sim_vectors = storage.get_db(repo).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )
    logger.debug(f'Found {len(sim_vectors)} snippets from sim search. top_k={sim_top_k}')

    start = time.time()
    ranks = reranker.rerank(search_query, sim_vectors["documents"][0], client)
    logger.info(f'Got new ranks from reranker, took {round(time.time() - start, 2)}s')

    documents = []
    for content, metadata in zip(
            sim_vectors["documents"][0],
            sim_vectors["metadatas"][0]):
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    # Select only the top ranked documents
    ranked_snippets = []
    ranks = await ranks
    for rank in ranks:
        ranked_snippets.append(
            documents[int(rank['corpus_id'])],
        )

    return format_context(ranked_snippets)
