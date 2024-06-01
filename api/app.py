import logging
import os
import time
from typing import List

import httpx
from fastapi import FastAPI, HTTPException, Depends
from httpx import AsyncClient
from langchain_core.documents import Document

from libs import storage
from libs.models import RequestData
from libs.proxies import embeddings, reranker, perform
from libs.proxies.chat import format_context, ChatWithRepo

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

collection_name = os.environ['CHROMA_COLLECTION']

app = FastAPI()


async def get_client():
    async with httpx.AsyncClient() as client:
        yield client


@app.post("/chat/")
async def chat_with_repo(request: RequestData, client: AsyncClient = Depends(get_client)):
    try:
        question = request.messages[0].content

        context = await build_context(
            search_query=question,
            sim_top_k=50,
            client=client
        )

        question_task = ChatWithRepo(
            question=question,
            context=context
        )

        logger.debug(f"Context generated:\n{context}")

        response = await perform(question_task, client)

        return response
    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")


async def build_context(
        search_query: str,
        sim_top_k: int,
        client: httpx.AsyncClient
) -> List:
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(search_query, client)
    logger.info(f'Got embedding for user search query, took {round(time.time() - start, 2)}s')

    logger.info(f'Searching collection {collection_name}')
    sim_vectors = storage.get_db(collection_name).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )

    start = time.time()
    ranks = reranker.rerank(search_query, sim_vectors["documents"][0], client)
    logger.info(f'Got new ranks from reranker, took {round(time.time() - start, 2)}s')
    logger.debug(f'Found {len(sim_vectors)} snippets from sim search. top_k={sim_top_k}')

    documents = []
    for content, metadata in zip(
            sim_vectors["documents"][0],
            sim_vectors["metadatas"][0]):
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    ranked_pairs = []
    ranks = await ranks
    for index in range(len(ranks)):
        ranked_pairs.append((
            documents[index],
            ranks[index]
        ))

    sorted_snippets = [
        pair[0] for pair in
        sorted(ranked_pairs, key=lambda x: x[1], reverse=True)
    ]

    return format_context(sorted_snippets)
