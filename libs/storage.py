import os

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

chroma_client = chromadb.HttpClient(
    host=os.environ['CHROMA_HOST'], port=int(os.environ['CHROMA_PORT']),
    settings=Settings(allow_reset=True, anonymized_telemetry=False))

vector_db = Chroma(
    embedding_function=OpenAIEmbeddings(disallowed_special=()),
    client=chroma_client
)
