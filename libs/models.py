import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Repo(BaseModel):
    name: str
    branch: str
    url: str
    documents: List
    tree: str
    metadata: Dict = {}


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class Choice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str]


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage]


class Object(BaseModel):
    object: str
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str
    data: List[Object]
    model: str
    usage: Optional[EmbeddingUsage]


class QueryPrompts(BaseModel):
    user: str
    system: str

    def api_format(self):

        return [
            {
                'role': 'system',
                'content': self.system
            },
            {
                'role': 'user',
                'content': self.user
            }
        ]


class Provider(BaseModel):
    name: str
    headers: Dict
    url: str


class Model(BaseModel):
    name: str
    provider: Provider
    endpoint: str

    @property
    def url(self):
        return f'{self.provider.url}/{self.endpoint}'


def flatten_list(original):
    return original[0]


def to_snippets(distances, documents, ids, metadatas, **_):
    """Convert a stream of vecdb lists into individual snippets"""
    snippets = []

    for distance, document, vecdb_idx, metadata in zip(
            distances[0],
            documents[0],
            ids[0],
            metadatas[0]
    ):
        metadata['vecdb_idx'] = vecdb_idx
        snippets.append(Document(
            vecdb_idx=vecdb_idx,
            metadata=metadata,
            distance=distance,
            page_content=document
        ))

    return snippets


class Message(BaseModel):
    role: str = 'user'
    content: Any


class RequestData(BaseModel):
    model: str = 'model_name'
    messages: List[Message]


class ProxyLLMTask:
    system_prompt = ''
    user_prompt = ''
    model = None

    def __init__(self, **kwargs):
        system_prompt = self.system_prompt.format(**kwargs)
        user_prompt = self.user_prompt.format(**kwargs)

        self._prompts = QueryPrompts(system=system_prompt, user=user_prompt)

    @property
    def prompts(self):
        return self._prompts
