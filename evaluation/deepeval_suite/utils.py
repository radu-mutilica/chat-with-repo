import json

from deepeval.synthesizer import doc_chunker
from libs.models import LLMResponseChunk


def clean(text):
    for line in text.splitlines():
        if line:
            line = line.strip()[6:]
            try:
                yield json.loads(line)
            except json.decoder.JSONDecodeError:
                print("Failed for text: {}".format(line))
                if not line or 'DONE' in text:
                    pass
                else:
                    raise


async def consume_stream(stream):
    content = ''
    async for message in await stream:
        for chunk in clean(message):
            content += LLMResponseChunk.parse_obj(chunk).raw

    return content


class RAGChunker(doc_chunker.DocumentChunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
