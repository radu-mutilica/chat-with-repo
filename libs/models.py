from typing import List, Optional, Dict

from langchain_core.documents import Document
from pydantic.v1 import BaseModel


class Repo(BaseModel):
    name: str
    branch: str
    url: str
    documents: List[Document]
    tree: str
    metadata: Dict = {}


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str]


class ProxyResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage]


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
