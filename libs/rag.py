import logging
import os
import time
from typing import List

import httpx
from httpx import AsyncClient
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from libs import storage, crawl_targets
from libs.models import RAGDocument, Message, RAGResponse
from libs.proxies import reranker, embeddings, perform_task, rephraser, stream_task
from libs.proxies.chat import format_context, ChatWithRepo

long_context_reorder = LongContextReorder()
logger = logging.getLogger(__name__)
sim_search_top_k = int(os.environ['SIM_SEARCH_TOP_K'])


async def context_pipeline(
        query: str,
        collection: str,
        sim_top_k: int,
        client: httpx.AsyncClient) -> List[Document]:
    """Main logic for building the RAG context.

    The pipeline consists of the following steps:
        1. Perform a similarity search using the query.
        2. Send documents to reranker.
        3. Reorder for "long context".
        4. Format into a {context} str to insert into the prompt.

    Args:
        query: (str) the user's search query.
        collection: (str) the vector db collection name.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        A list of RAGDocument objects.
    """

    logger.info(f'Searching collection {collection}')

    sim_vectors = await sim_search(query, collection, sim_top_k, client)

    scores = reranker.rerank(query, sim_vectors['documents'][0], client)
    # Build a list with only the ranked documents, these are the final ones that
    # are used for formatting the RAG context
    documents = []
    for content, metadata in zip(sim_vectors['documents'][0], sim_vectors['metadatas'][0]):
        documents.append(RAGDocument(page_content=content, metadata=metadata))

    for index, score in enumerate(await scores):
        documents[index].metadata['score'] = score

    ranked_documents = sorted(documents, key=lambda d: d.metadata['score'], reverse=True)

    # Final touches, apply a long context reorder to mitigate the "lost in the middle" effect
    ordered_documents = long_context_reorder.transform_documents(ranked_documents)

    return list(RAGDocument.parse_obj(doc) for doc in ordered_documents)


async def sim_search(query: str, collection: str, sim_top_k: int, client: httpx.AsyncClient):
    """Perform a similarity search on the database and find top_k related vectors.

    Args:
        query: (str) the user's search query.
        collection: (str) the vector db collection name.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        Top similar vectors.
    """
    start = time.time()
    search_query_embedding = await embeddings.generate_embedding(query, client)
    logger.info(f'Embedding took {time.time() - start:.2f}s')

    start = time.time()
    sim_vectors = storage.get_db(collection).query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )
    logger.info(f'Simsearch took {time.time() - start:.2f}s')

    return sim_vectors


async def answer_query(
        last_message: Message,
        chat_history: List[Message],
        client: AsyncClient) -> RAGResponse:
    """Do some query processing first, check if there is any chat history, create
    an llm prompt based on the previous facts and send it over to the llm.

    Args:
        last_message (Message): the last message aka the user query.
        chat_history (List[Message]): the chat history, might be an empty list.
        client: (httpx.AsyncClient) the client.

    Raises:
        AssertionError: if we have no info or crawl data about the repo at hand.

    Returns:
        A RAGResponse object that includes the stream and the provided rag context.
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
    context = await context_pipeline(
        collection=crawl_targets[subnet]['target_collection'],
        query=rag_query,
        sim_top_k=sim_search_top_k,
        client=client
    )
    formatted_context = format_context(context)
    logger.info(f"{time.time() - start:.2f}s: Context size={len(formatted_context)}:"
                f"\n{formatted_context}")

    chat_with_repo_task = ChatWithRepo(
        question=rag_query,
        context=formatted_context,
        github_name=crawl_targets[subnet]['name'],
        repo_name=subnet

    )
    # Hardcode this to stream back the response for now
    return RAGResponse(stream=stream_task(chat_with_repo_task), context=context)
