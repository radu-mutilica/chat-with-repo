import config

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vector_db = Chroma(
    # todo: what is disallowed_special actually doing?
    embedding_function=OpenAIEmbeddings(disallowed_special=()),
    persist_directory=config.chroma_persist_directory
)