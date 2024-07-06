import asyncio
import os
from asyncio import Task
from typing import List

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from libs import storage, crawl_targets
from libs.http import OptimizedAsyncClient
from libs.models import RAGDocument, Message, RAGResponse
from libs.proxies import reranker, embeddings, perform_task, rephraser, stream_task
from libs.proxies.chat import format_context, ChatWithRepo

long_context_reorder = LongContextReorder()
sim_search_top_k = int(os.environ['SIM_SEARCH_TOP_K'])


async def context_pipeline(
        query: str,
        sim_top_k: int,
        client: OptimizedAsyncClient,
        vector_db_future: Task,
        # todo: refactor these warmup tasks into a sort of warmup controller for all the endpoints
        reranker_warm_up_future: Task = None,
        embeddings_warm_up_task: Task = None,
) -> List[Document]:
    """Main logic for building the RAG context.

    The pipeline consists of the following steps:
        1. Perform a similarity search using the query.
        2. Send documents to reranker.
        3. Reorder for "long context".
        4. Format into a {context} str to insert into the prompt.

    Args:
        query: (str) the user's search query.
        sim_top_k: (int) the top_k for the sim search.
        client: (httpx.AsyncClient) the httpx client.
        vector_db_future: (Coroutine) the vector db.
        reranker_warm_up_future: (Task) if passed in, will await it.
        embeddings_warm_up_task: (Task) if passed in, will await it.

    Returns:
        A list of RAGDocument objects.
    """
    if embeddings_warm_up_task:
        embeddings_warm_up_task.result()

    sim_vectors = await sim_search(query, sim_top_k, vector_db_future, client)
    docs, metas = sim_vectors['documents'][0], sim_vectors['metadatas'][0]

    # Perform a doc reranking step, feeding in the similar vectors, returning the result of
    # Cross-Encoding them, with a pretty low top_k
    # This _should_ establish the SSL handshake before the main reranker call
    if reranker_warm_up_future:
        reranker_warm_up_future.result()

    reranker_response = reranker.rerank(query, docs, client)

    ranked_documents = []
    # Ranks are returned in order, which makes this easy
    for rank in await reranker_response:
        ranked_documents.append(
            RAGDocument(
                # the `corpus_id` is the document index
                page_content=docs[rank['corpus_id']],
                metadata=metas[rank['corpus_id']]
            ))

    # Final touches, apply a long context reorder to mitigate the "lost in the middle" effect
    ordered_documents = long_context_reorder.transform_documents(ranked_documents)

    return list(ordered_documents)


async def sim_search(
        query: str,
        sim_top_k: int,
        vector_db: Task,
        client: OptimizedAsyncClient):
    """Perform a similarity search on the database and find top_k related vectors.

    Args:
        query: (str) the user's search query.
        sim_top_k: (int) the top_k for the sim search.
        vector_db: (Task) the vector db.
        client: (httpx.AsyncClient) the httpx client.

    Returns:
        Top similar vectors.
    """
    search_query_embedding = await embeddings.generate_embedding([query], client)

    sim_vectors = vector_db.result().query(
        query_embeddings=search_query_embedding,
        n_results=sim_top_k
    )

    return sim_vectors


async def answer_query(
        last_message: Message,
        chat_history: List[Message],
        client: OptimizedAsyncClient) -> RAGResponse:
    """Do some query processing first, check if there is any chat history, create
    an llm prompt based on the previous facts and send it over to the llm.

    Args:
        last_message (Message): the last message aka the user query.
        chat_history (List[Message]): the chat history, might be an empty list.
        client: (OptimizedAsyncClient) the client.

    Raises:
        AssertionError: if we have no info or crawl data about the repo at hand.

    Returns:
        A RAGResponse object that includes the stream and the provided rag context.
    """
    # todo: Assume all messages are about the same repo
    query, subnet = last_message.content.query, last_message.content.repo
    assert subnet in crawl_targets, 'Not a valid repo'

    async with asyncio.TaskGroup() as tg:
        # Check for a chat history, and if present, rephrase the query given the history.
        # This step is important to guarantee good simsearch results further down
        if chat_history:
            rephrased_query = tg.create_task(perform_task(
                rephraser.RephraseGivenHistory(
                    query=query,
                    chat_history=chat_history),
                client=client))

        reranker_warm_up_task = tg.create_task(client.warmup_if_needed(
            reranker.model.url,
            reranker.model.provider.headers
        ))

        embeddings_warm_up_task = tg.create_task(client.warmup_if_needed(
            embeddings.model.url,
            embeddings.model.provider.headers
        ))

        vector_db_task = tg.create_task(
            storage.get_db(crawl_targets[subnet]['target_collection'])
        )

    context = await context_pipeline(
        query=query if not chat_history else rephrased_query.result(),
        sim_top_k=sim_search_top_k,
        client=client,
        vector_db_future=vector_db_task,
        reranker_warm_up_future=reranker_warm_up_task,
        embeddings_warm_up_task=embeddings_warm_up_task
    )

    formatted_context = format_context(context)

    chat_with_repo_task = ChatWithRepo(
        question=query,
        context=formatted_context,
        github_name=crawl_targets[subnet]['name'],
        repo_name=subnet
    )

    # Hardcode this to stream back the response for now
    return RAGResponse(stream=stream_task(chat_with_repo_task), context=context)
