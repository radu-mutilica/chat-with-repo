from typing import List

import httpx

from libs.models import EmbeddingResponse, Model
from libs.proxies.providers import openai

embeddings = Model(name='text-embedding-3-small', provider=openai, endpoint='embeddings')


async def generate_embedding(search_string: str, client: httpx.AsyncClient) -> List[float]:
    payload = {
        "model": embeddings.name,
        "input": [search_string]
    }

    response = await client.post(
        url=embeddings.url,
        json=payload,
        headers=embeddings.provider.headers)

    response.raise_for_status()
    response = EmbeddingResponse(**response.json())

    return response.data[0].embedding
