import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from httpx import AsyncClient
from langchain_core.documents import Document
from starlette.responses import StreamingResponse

from libs import storage
from libs.models import RequestData, ChatQuery
from libs.proxies import embeddings, reranker, stream_task, perform_task
from libs.proxies.chat import format_context, ChatWithRepo
from libs.proxies.query_expansion import QueryExpander
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


async def get_client():
    """Helper func to keep a client hot"""
    async with httpx.AsyncClient() as client:
        yield client


@app.post("/chat/", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def chat_with_repo(request: RequestData, client: AsyncClient = Depends(get_client)):
    """Endpoint for chatting with your repo.

    Get the user's search string, build the context, format the prompt and issue the LLM call.

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

        # Commented this out since the model used for answering the question is
        # pretty good at detecting 'erroneous' user input
        #
        # query_inspector_task = QueryInspector(
        #     user_input=query
        # )
        # query_inspector_response = await perform_task(task=query_inspector_task, client=client)
        # assert query_inspector_response.strip() == 'TRUE', 'Invalid user query'
        start = time.time()
        # expanded_queries = await perform_task(QueryExpander(
        #     question=query
        # ), client=client)

        # delta_seconds = time.time() - start
        # logger.info(f"{round(delta_seconds, 2)}s: Query expanded")

        expanded_query = ChatQuery(
            main=query,
            expansions=[]  # no variants atm, still investigating their usefulness
        )
        start = time.time()
        context = await build_rag_context(
            queries=expanded_query,
            sim_top_k=sim_search_top_k,
            client=client
        )
        delta_seconds = time.time() - start
        logger.info(f"{round(delta_seconds, 2)}s: Context length={len(context)}:\n{context}")

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


# todo: refactor this big boy
async def build_rag_context(
        queries: ChatQuery,
        sim_top_k: int,
        client: httpx.AsyncClient) -> str:
    """Main logic for the RAG component.

    Use similarity search to gather a list of documents, then rerank them and format them into
    a prompt template.

    Args:
        queries: (ChatQuery) the user's search queries (expanded).
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A string representing the context for the final llm prompt.
    """

    logger.info(f'Searching collection {collection_name}')

    all_vectors = await sim_search(queries, sim_top_k, client)

    start = time.time()
    ranks = reranker.rerank(queries.main, all_vectors['documents'][0], client)
    logger.info(f'Got new ranks from reranker, took {round(time.time() - start, 2)}s')

    # Build a list with only the ranked documents, these are the final ones that
    # are used for formatting the RAG context
    documents = []
    for content, metadata in zip(
            all_vectors['documents'][0],
            all_vectors['metadatas'][0]):
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    ranked_snippets = []
    ranks = await ranks
    for rank in ranks:
        ranked_snippets.append(
            documents[int(rank['corpus_id'])],
        )

    return format_context(ranked_snippets)


async def sim_search(queries: ChatQuery, sim_top_k: int, client: httpx.AsyncClient):
    return await __sim_search(queries.main, sim_top_k, client)


async def __sim_search(query: str, sim_top_k: int, client: httpx.AsyncClient) -> Dict:
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(query, client)
    logger.info(f'Embedding took {round(time.time() - start, 2)}s')
    sim_vectors = storage.get_db(collection_name).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )
    logger.debug(
        f'Got {len(sim_vectors["documents"][0])} top_k={sim_top_k}')
    return sim_vectors
