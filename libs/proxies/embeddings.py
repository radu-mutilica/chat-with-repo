import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from libs.http import OptimizedAsyncClient
from libs.models import Model
from libs.proxies.providers import hf_embeddings

model = Model(name='', provider=hf_embeddings, endpoint='')

logger = logging.getLogger(__name__)


async def generate_embedding(
        documents: List[str],
        client: OptimizedAsyncClient) -> List[List[float]]:
    """Generate one or more embeddings from a list of documents.

    Args:
        documents (List[str]): list of documents to compute embeddings for.
        client: (OptimizedAsyncClient): client to use for asynchronous requests.

    Returns:
        A list of embeddings for each document.
    """
    payload = {
        "inputs": documents
    }

    response = await client.post(
        url=model.url,
        json=payload,
        headers=model.provider.headers)

    response.raise_for_status()

    return response.json()


# noinspection PyShadowingBuiltins,PyProtocol
class HFEmbeddingFunc(EmbeddingFunction[Documents]):
    def __init__(self, client: OptimizedAsyncClient):
        """Custom HF embedding function, passed to ChromaDB.

        This embedding function uses a dedicated HF endpoint to generate embeddings.

        Args:
            client: (OptimizedAsyncClient): client to use for asynchronous requests.
        """
        self.client = client
        self.batch_size = 5
        # Unfortunately, have to rely on an extra thread to do the requests to the endpoint
        # since we're already inside a running event loop, and within a 'sync' context. This
        # code is executed by the ChromaDB ORM
        self.executor = ThreadPoolExecutor(max_workers=1)

    def __call__(self, input: Documents) -> Embeddings:
        """This is the method that Chroma calls"""
        return self._run_async(self.batch_documents, input)

    def _run_async(self, async_func, *args):
        """Wrapper function for running an async func in a separate thread."""
        def wrapper():
            asyncio.set_event_loop(asyncio.new_event_loop())
            return asyncio.get_event_loop().run_until_complete(async_func(*args))

        return self.executor.submit(wrapper).result()

    async def batch_documents(self, input: Documents) -> Embeddings:
        """Helper func to split the documents into batches, otherwise we exceed the
        http payload limit"""
        all_embeddings = []

        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            all_embeddings.extend(await generate_embedding(batch, self.client))

        return all_embeddings

    def __del__(self):
        self.executor.shutdown(wait=False)
