import os
from typing import List

from httpx import AsyncClient

RERANKER_HEADERS = {
    "Content-Type": "application/json",
}


async def rerank(query: str, documents: List[str], client: AsyncClient):
    """Helper function to build a payload and rerank some documents"""
    payload = {
        "model": "crossencoder",
        "messages": [
            {
                "role": "system",
                "content": {
                    "query": query,
                    "documents": documents
                }
            }

        ]
    }

    response = await client.post(
        f'http://{os.environ["RERANKER_HOST"]}:{os.environ["RERANKER_PORT"]}/rerank',
        json=payload,
        headers=RERANKER_HEADERS
    )

    response.raise_for_status()

    return response.json()
