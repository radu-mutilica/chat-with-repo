import httpx

from libs.models import ProxyLLMTask, CompletionResponse

timeout = httpx.Timeout(10.0, read=None)


async def perform(task: ProxyLLMTask, client: httpx.AsyncClient) -> CompletionResponse:
    payload = {
        "model": task.model.name,
        "messages": task.prompts.api_format()
    }

    response = await client.post(
        url=task.model.url,
        json=payload,
        headers=task.model.provider.headers,
        timeout=timeout)

    response.raise_for_status()
    raw_response = response.json()
    return raw_response['choices'][0]['message']['content']
