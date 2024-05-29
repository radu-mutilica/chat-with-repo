import httpx

import config
from libs.models import ProxyResponse


async def generate_embedding(search_string: str, client: httpx.AsyncClient) -> ProxyResponse:
    payload = {
        "model": config.embeddings.name,
        "input": [search_string]
    }

    response = await client.post(
        url=config.embeddings.url,
        json=payload,
        headers=config.embeddings.provider.headers)

    response.raise_for_status()

    return ProxyResponse(**response.json())
