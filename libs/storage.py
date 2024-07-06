import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

emb_fn = OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name='text-embedding-3-small'
)

vectordb = chromadb.HttpClient(
    host=os.environ['CHROMA_HOST'],
    port=int(os.environ['CHROMA_PORT']),
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)


async def get_db(collection):
    return vectordb.get_collection(
        name=collection,
        embedding_function=emb_fn
    )
