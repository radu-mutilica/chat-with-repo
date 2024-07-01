import logging
from typing import AsyncGenerator, Union

import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, \
    retry_if_result, RetryError

from libs.models import ProxyLLMTask

logger = logging.getLogger(__name__)
rate_limiter = AsyncLimiter(3, 2)
timeout = httpx.Timeout(20, read=None)


class EmptyLLMResponse(Exception):
    """Raised when the LLM API response is empty (typically empty list)"""


def log_error(retry_state):
    # error handler
    logger.exception(
        "Max retries exceeded",
        extra={"stats": retry_state.retry_object.statistics},
    )


@retry(
    retry=(
            retry_if_exception_type(
                Union[
                    httpx.TimeoutException,
                    httpx.HTTPStatusError,
                    EmptyLLMResponse,
                    httpx.RemoteProtocolError]) |
            retry_if_result(lambda response: response.status_code in [524, 429])
    ),
    stop=stop_after_attempt(5),
    wait=wait_fixed(60),
    retry_error_callback=log_error
)
async def __make_request(url, payload, headers, client):
    """Helper func to do a generic request, wrapped with retry handler"""
    async with rate_limiter:
        # todo: test if need new client here or not
        response = await httpx.AsyncClient().post(
            url=url,
            json=payload,
            headers=headers,
            timeout=timeout
        )

    response.raise_for_status()

    if not response.json():
        raise EmptyLLMResponse()

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

    try:
        response = await __make_request(
            url=task.model.url,
            payload=payload,
            headers=task.model.provider.headers,
            client=client
        )
        logger.info(f'Response: {response}')
        data = response.json()
        logger.info(f'Data: {data}')

    except RetryError:
        logger.error(f"RetryError for {task.model.url}")
        logger.error(f"Payload: \n{payload}")
        for idx, message in enumerate(payload['messages']):
            logger.error(f'Msg #{idx}: size of {message["role"]} '
                         f'prompt: {len(message["content"])}')
        raise
    else:
        try:
            content = data[0]['choices'][0]['delta']['content']
        finally:
            for idx, message in enumerate(payload['messages']):
                logger.debug(
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

    async with httpx.AsyncClient().stream(
            'POST',
            url=task.model.url,
            json=payload,
            headers=task.model.provider.headers,
            timeout=timeout) as r:
        async for chunk in r.aiter_text():
            yield chunk
