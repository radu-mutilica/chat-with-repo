import time

import httpx
from langchain_community.document_transformers import LongContextReorder

from api.app import logger
from libs import storage
from libs.models import RAGDocument
from libs.proxies import reranker, embeddings
from libs.proxies.chat import format_context

long_context_reorder = LongContextReorder()


async def context_pipeline(
        query: str,
        collection: str,
        sim_top_k: int,
        client: httpx.AsyncClient) -> str:
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
        A str which is the formatted context.
    """

    logger.info(f'Searching collection {collection}')

    sim_vectors = await sim_search(query, collection, sim_top_k, client)

    start = time.time()
    ranks = reranker.rerank(query, sim_vectors['documents'][0], client)
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
