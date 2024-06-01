import os
from typing import List

from httpx import AsyncClient
from langchain_core.documents import Document

RERANKER_HEADERS = {
    "Content-Type": "application/json",
}


async def rerank(query: str, snippets: List[Document], client: AsyncClient):
    payload = {
        "model": "crossencoder",
        "messages": [
            {
                "role": "system",
                "content": {
                    "query": query,
                    "documents": [snippet.dict() for snippet in snippets]
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
