import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Depends
from httpx import AsyncClient
from langchain_core.documents import Document
from starlette.responses import StreamingResponse

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

collection_name = os.environ['CHROMA_COLLECTION']
sim_search_top_k = int(os.environ['SIM_SEARCH_TOP_K'])

app = FastAPI()


async def get_client():
    """Helper func to keep a client hot"""
    async with httpx.AsyncClient() as client:
        yield client


@app.post("/chat/")
async def chat_with_repo(request: RequestData, client: AsyncClient = Depends(get_client)):
    """Endpoint for chatting with your repo.

    Get the user's search string, build the context, format the prompt and issue the LLM call.

    Args:
        request: (RequestData) the request.
        client: (httpx.AsyncClient) the client.
    """
    try:
        query = request.messages[0].content

        # Commented this out since the model used for answering the question is
        # pretty good at detecting 'erroneous' user input
        #
        # query_inspector_task = QueryInspector(
        #     user_input=query
        # )
        # query_inspector_response = await perform_task(task=query_inspector_task, client=client)
        # assert query_inspector_response.strip() == 'TRUE', 'Invalid user query'

        context = await build_rag_context(
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

    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")


async def build_rag_context(
        search_query: str,
        sim_top_k: int,
        client: httpx.AsyncClient) -> str:
    """Main logic for the RAG component.

    Use similarity search to gather a list of documents, then rerank them and format them into
    a prompt template.

    Args:
        search_query: (str) the user's search query.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A string representing the context for the final llm prompt.
    """
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(search_query, client)
    logger.info(f'Got embedding for user search query, took {round(time.time() - start, 2)}s')

    logger.info(f'Searching collection {collection_name}')
    sim_vectors = storage.get_db(collection_name).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )
    logger.debug(f'Found {len(sim_vectors)} snippets from sim search. top_k={sim_top_k}')

    # # Filter out any duplicated code snippets (if we already embed the parent file, no need
    # # to also embed and repeat any of its code snippets)
    # referenced_files = [
    #     m['file_path'] for m in sim_vectors['metadatas'][0]
    #     if m['vecdb_idx'].endswith('main')
    # ]
    #
    # filtered_metadatas, filtered_documents = [], []
    # skipped_docs = 0
    # for meta, doc in zip(sim_vectors['metadatas'][0], sim_vectors['documents'][0]):
    #     if not meta['vecdb_idx'].endswith('main') and meta['file_path'] not in referenced_files:
    #         filtered_metadatas.append(meta)
    #
    #         filtered_documents.append(doc)
    #     else:
    #         skipped_docs += 1
    #         logger.debug(f'Skipping metadata={meta} in context. Already referenced entire file')
    #
    # if skipped_docs:
    #     logger.debug(f'Skipped {skipped_docs} documents to deduplicate context')

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
