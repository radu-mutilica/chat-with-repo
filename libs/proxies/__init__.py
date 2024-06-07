import logging
from typing import AsyncGenerator, Union

import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, retry_if_result

from libs.models import ProxyLLMTask

logger = logging.getLogger(__name__)
timeout = httpx.Timeout(20, read=None)
rate_limiter = AsyncLimiter(1, 1)


@retry(
    retry=(
            retry_if_exception_type(
                Union[httpx.TimeoutException, httpx.HTTPStatusError, AssertionError]) |
            retry_if_result(lambda response: response.status_code in [524, 429])
    ),
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
)
async def __make_request(url, payload, headers, client):
    """Helper func to do a generic request, wrapped with retry handler"""
    response = await client.post(
        url=url,
        json=payload,
        headers=headers,
        timeout=timeout
    )
    response.raise_for_status()
    assert response.json(), 'Response is empty'

    return response


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
            response = await __make_request(
                url=task.model.url,
                payload=payload,
                headers=task.model.provider.headers,
                client=client
            )
            raw_response = response.json()
            logger.info(f'Response: {raw_response}')

            assert raw_response, "Response is empty"

        except httpx.HTTPError as exc:
            logger.error(f"HTTP Exception for {exc.request.url}")
            logger.error(exc.response.text)
            logger.debug(f"Payload: \n{payload}")
            for idx, message in enumerate(payload['messages']):
                logger.debug(f'Msg #{idx}: size of {message["role"]} '
                             f'prompt: {len(message["content"])}')
            raise
        else:
            try:
                content = raw_response[0]['choices'][0]['delta']['content']
            finally:
                for idx, message in enumerate(payload['messages']):
                    logger.info(
                        f'Msg #{idx}: size of {message["role"]} prompt: {len(message["content"])}')

            if task.post_processing_func:
                content = task.post_processing_func(content)

            return content


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


# async def run_task(task: ProxyLLMTask) -> str:
#     """todo: remove this after refactoring"""
#     payload = {
#         "model": task.model.name,
#         "messages": task.prompts.api_format()
#     }
#     if task.extra_settings:
#         payload.update(task.extra_settings)
#
#     # Send the request and stream the response
#     client = httpx.AsyncClient()
#     response = await client.post(
#         url=task.model.url,
#         json=payload,
#         headers=task.model.provider.headers,
#         timeout=timeout
#     )
#     return response.json()[0]['choices'][0]['delta']['content']
