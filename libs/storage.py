import os

import chromadb
from chromadb.api.models import Collection
from chromadb.config import Settings

from libs.http import OptimizedAsyncClient
from libs.proxies.embeddings import HFEmbeddingFunc

vector_db = chromadb.HttpClient(
    host=os.environ['CHROMA_HOST'],
    port=int(os.environ['CHROMA_PORT']),
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)


async def get_db(collection, client: OptimizedAsyncClient) -> Collection:
    """Get a ChromaDB collection by name.

    Note: The client passed in is used in the EmbeddingFunction.

    Args:
        collection: (str) the name of the collection to get.
        client: (OptimizedAsyncClient) the client to use for generating embeddings via
        async http.

    Returns:
        A ChromaDB collection.
    """
    return vector_db.get_collection(
        name=collection,
        embedding_function=HFEmbeddingFunc(client)
    )
