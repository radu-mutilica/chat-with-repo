from typing import List

from libs.http import OptimizedAsyncClient
from libs.models import Model
from libs.proxies.providers import hf_reranker

model = Model(name='reranker', provider=hf_reranker, endpoint='')


async def rerank(query: str, documents: List[str], client: OptimizedAsyncClient):
    """Helper function to create all crawling tasks (one per repo defined in the yaml file)"""
    payload = {
        'query': query,
        'documents': documents
    }

    response = await client.post(
        url=model.url,
        json={'inputs': payload},
        headers=model.provider.headers)

    response.raise_for_status()

    return response.json()
