import logging
import os
from typing import List

logger = logging.getLogger(__name__)
reranker_headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {os.environ["HF_API_KEY"]}',
    'Content-Type': 'application/json'
}
reranker_api_url = os.environ['RERANKER_API']


async def rerank(query: str, documents: List[str], client) -> List[float]:
    """Helper function to create all crawling tasks (one per repo defined in the yaml file)"""
    payload = {
        'query': query,
        'documents': documents

    }

    response = await client.post(
        reranker_api_url,
        json={'inputs': payload},
        headers=reranker_headers
    )
    response.raise_for_status()

    return response.json()
