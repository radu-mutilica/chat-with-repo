import json
from typing import AsyncGenerator

from deepeval.synthesizer import doc_chunker
from libs.models import LLMResponseChunk


def clean(text) -> str:
    """Helper func to sanitize chunks of the response from the LLM"""
    for line in text.splitlines():
        if line:
            line = line[6:]
            try:
                yield json.loads(line)
            except json.decoder.JSONDecodeError:
                # todo: what do here?
                print("Failed for text: {}".format(line))


async def consume_stream(stream: AsyncGenerator) -> str:
    """Helper func to consume a stream of text"""
    content = ''
    async for message in stream:
        for chunk in clean(message):
            content += LLMResponseChunk.parse_obj(chunk).raw

    return content


class RAGChunker(doc_chunker.DocumentChunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
