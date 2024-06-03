import logging
from typing import AsyncGenerator

import httpx

from libs.models import ProxyLLMTask

timeout = httpx.Timeout(15.0, read=None)

logger = logging.getLogger(__name__)


async def perform_task(task: ProxyLLMTask, client: httpx.AsyncClient) -> str:
    """Prepare a payload for an llm task, fire it and return back the response.

    Args:
        task: (ProxyLLMTask) an llm task to perform.
        client: (httpx.AsyncClient) a httpx client.

    Returns:
        The str result of the task.
    """
    payload = {
        "model": task.model.name,
        "messages": task.prompts.api_format()
    }
    try:
        response = await client.post(
            url=task.model.url,
            json=payload,
            headers=task.model.provider.headers,
            timeout=timeout)

        response.raise_for_status()
        raw_response = response.json()

        return raw_response['choices'][0]['message']['content']
    except httpx.HTTPError as exc:
        logger.error(f"HTTP Exception for {exc.request.url} - {exc}")
        logger.debug(f"Payload: \n{payload}")
        for idx, message in enumerate(payload['messages']):
            logger.debug(f'Msg #{idx}: size of {message["role"]} prompt: {len(message["content"])}')
        raise


async def stream_task(task: ProxyLLMTask) -> AsyncGenerator:
    """Prepare a payload for an llm task, fire it and stream back the response.

    Args:
        task: (ProxyLLMTask) the task to issue to the llm.

    Returns:
        AsyncGenerator: the response stream.
    """
    payload = {
        "model": task.model.name,
        "messages": task.prompts.api_format()
    }

    # Send the request and stream the response
    async with httpx.AsyncClient().stream(
            'POST',
            url=task.model.url,
            json=payload,
            headers=task.model.provider.headers,
            timeout=timeout) as r:
        async for chunk in r.aiter_bytes():
            yield chunk
