import asyncio
import logging
from typing import AsyncGenerator

import httpx
from aiolimiter import AsyncLimiter

from libs.models import ProxyLLMTask

logger = logging.getLogger(__name__)
timeout = httpx.Timeout(20, read=None)
rate_limiter = AsyncLimiter(3, 1)


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

    if task.extra_settings:
        payload.update(task.extra_settings)

    async with rate_limiter:
        try:
            response = await client.post(
                url=task.model.url,
                json=payload,
                headers=task.model.provider.headers,
                timeout=timeout
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)

                    logger.error(f"Rate limit exceeded. Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:  # If no Retry-After header, wait a default time
                    logger.error("Rate limit exceeded. Retrying after 60 seconds...")
                    await asyncio.sleep(60)

                response = await client.post(
                    url=task.model.url,
                    json=payload,
                    headers=task.model.provider.headers,
                    timeout=timeout
                )

                response.raise_for_status()
            else:
                response.raise_for_status()

        except httpx.HTTPError as exc:
            logger.error(f"HTTP Exception for {exc.request.url} - {exc}")
            logger.debug(f"Payload: \n{payload}")
            for idx, message in enumerate(payload['messages']):
                logger.debug(f'Msg #{idx}: size of {message["role"]} '
                             f'prompt: {len(message["content"])}')
            raise
        else:

            raw_response = response.json()
            logger.info(f'Response: {raw_response}')
            return response.json()[0]['choices'][0]['delta']['content']


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