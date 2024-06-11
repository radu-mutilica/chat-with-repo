import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from httpx import AsyncClient
from langchain_community.document_transformers import LongContextReorder
from starlette.responses import StreamingResponse

from libs import storage
from libs.models import RequestData, ChatQuery, RAGDocument
from libs.proxies import embeddings, reranker, stream_task
from libs.proxies.chat import format_context, ChatWithRepo
from libs.utils import register_profiling_middleware

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

collection_name = os.environ['CHROMA_COLLECTION']
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
long_context_reorder = LongContextReorder()


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
        content = request.messages[0].content
        query, repo = content.query, content.repo

        assert repo == 'subnet-19', 'Only able to answer questions about sn19 at the moment'
        # todo: when we get multi repo crawler out, get the github_repo from the crawl_targets
        github_repo = 'vision-4.0'

        expanded_query = ChatQuery(
            main=query,
            expansions=[]  # no variants atm, still investigating their usefulness
        )
        start = time.time()
        context = await rag_context_pipeline(
            queries=expanded_query,
            sim_top_k=sim_search_top_k,
            client=client
        )
        delta_seconds = time.time() - start
        logger.info(f"{delta_seconds:.2f}s: Context length={len(context)}:\n{context}")

        chat_with_repo_task = ChatWithRepo(
            question=expanded_query.main,
            context=context,
            github_name=github_repo,
            repo_name=repo
        )
        # Hardcode this to a streaming response. Once this model has support
        # for standard responses, we can fix this
        return StreamingResponse(stream_task(chat_with_repo_task))

    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")


async def rag_context_pipeline(
        queries: ChatQuery,
        sim_top_k: int,
        client: httpx.AsyncClient) -> str:
    """Main logic for building the RAG context.

    The pipeline consists of the following steps:
        1. Perform a similarity search using the query.
        2. Send documents to reranker.
        3. Reorder for "long context".
        4. Format into a {context} str to insert into the prompt.

    Args:
        queries: (ChatQuery) the user's search queries (expanded).
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A string representing the context for the final llm prompt.
    """

    logger.info(f'Searching collection {collection_name}')

    sim_vectors = await sim_search(queries, sim_top_k, client)

    start = time.time()
    ranks = reranker.rerank(queries.main, sim_vectors['documents'][0], client)
    logger.info(f'Got new ranks from reranker, took {time.time() - start:.2f}s')

    # Build a list with only the ranked documents, these are the final ones that
    # are used for formatting the RAG context
    documents = []
    for content, metadata in zip(sim_vectors['documents'][0], sim_vectors['metadatas'][0]):
        documents.append(RAGDocument(page_content=content, metadata=metadata))

    ranked_documents = []
    for rank in await ranks:
        ranked_documents.append(
            documents[int(rank['corpus_id'])],
        )

    # Final touches, apply a long context reorder to mitigate the "lost in the middle" effect
    ordered_documents = long_context_reorder.transform_documents(ranked_documents)

    return format_context(ordered_documents)


async def sim_search(queries: ChatQuery, sim_top_k: int, client: httpx.AsyncClient):
    """Perform a similarity search on the database and find top_k related vectors.

    Args:
        queries: (ChatQuery) the user's search queries.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        Top similar vectors.
    """
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(queries.main, client)
    logger.info(f'Embedding took {time.time() - start:.2f}s')

    start = time.time()
    sim_vectors = storage.get_db(collection_name).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )
    logger.info(f'Simsearch took {time.time() - start:.2f}s')

    return sim_vectors


