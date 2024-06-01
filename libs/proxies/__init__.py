import httpx

from libs.models import ProxyLLMTask, CompletionResponse

timeout = httpx.Timeout(10.0, read=None)

import logging

logger = logging.getLogger(__name__)


async def perform(task: ProxyLLMTask, client: httpx.AsyncClient) -> CompletionResponse:
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
        logger.debug(f"Info about payload: \n{payload}")
        for idx, message in enumerate(payload['messages']):
            logger.debug(f'Msg #{idx}: size of {message["role"]} prompt: {len(message["content"])}')
        raise

