import logging
from typing import List, Dict, AsyncGenerator, Any

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class Repo(BaseModel):
    name: str
    branch: str
    url: str
    documents: List
    tree: str
    summary: Dict = {}


class RepoOwner(BaseModel):
    id: int
    login: str
    avatar_url: str


class RepoBranch(BaseModel):
    name: str
    last_commit_ts: int


class RepoCrawlStats(BaseModel):
    repo_id: str  # internal database id
    tag: str
    added_ts: int | None = None  # todo: fix this
    github_id: int
    name: str
    full_name: str
    description: str | None = None  # this might be None
    owner: RepoOwner
    branch: RepoBranch


class RepoCrawlTarget(BaseModel):
    repo_id: str
    url: str
    branch: str
    name: str
    target_collection: str
    tag: str


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class Object(BaseModel):
    object: str
    index: int
    embedding: List[float]


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


class RankContent(BaseModel):
    query: str
    documents: List[str]


class ChatAnswer(BaseModel):
    answer: str
    repo: str

    @property
    def raw(self):
        return self.answer


class ChatQuery(BaseModel):
    query: str
    repo: str

    @property
    def raw(self):
        return self.query


class Message(BaseModel):
    role: str = 'user'
    content: ChatQuery | ChatAnswer | RankContent


class RequestData(BaseModel):
    model: str = 'model_name'
    messages: List[Message]

    @property
    def history(self):
        return self.messages[:-1]

    @property
    def last_message(self):
        return self.messages[-1]


class ProxyLLMTask:
    system_prompt = ''
    user_prompt = ''
    model = None
    extra_settings = None
    post_processing_func = None
    pre_processing_func = None

    def __init__(self, **kwargs):
        if self.pre_processing_func:
            kwargs = self.pre_processing_func(kwargs)

        system_prompt = self.system_prompt.format(**kwargs)
        user_prompt = self.user_prompt.format(**kwargs)

        self._prompts = QueryPrompts(system=system_prompt, user=user_prompt)

    @property
    def prompts(self):
        return self._prompts


class DocumentRank(BaseModel):
    corpus_id: int
    score: float


contextual_file_fmt = """
*** File: *** {path}
*** Summary: ***: {summary}"""

contextual_code_fmt = """
*** File: *** {path}
*** Code: *** 
{code}
*** Summary: ***: {summary}"""

code_fmt = """```{language}
{raw_code}
```"""


class RAGDocument(Document):
    def __str__(self):
        if self.metadata['document_type'] == 'file-summary':
            return contextual_file_fmt.format(
                path=self.metadata['file_path'],
                summary=self.page_content,
            )
        elif self.metadata['document_type'] == 'code-snippet':
            return contextual_code_fmt.format(
                path=self.metadata['file_path'],
                summary=self.page_content,
                code=code_fmt.format(
                    language=self.metadata['language'],
                    raw_code=self.metadata['original_page_content']
                )
            )


class ChunkContent(BaseModel):
    content: str


class ChunkMessage(BaseModel):
    delta: ChunkContent


class LLMResponseChunk(BaseModel):
    choices: List[ChunkMessage]

    @property
    def raw(self):
        return self.choices[0].delta.content


class RAGPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream: AsyncGenerator
    context: List[Any]
    formatted: str


class Rank(BaseModel):
    doc_idx: int
    score: float
