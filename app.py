import os
import tempfile
from typing import List

import httpx
import langchain
from directory_structure import Tree
from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from libs import logger, splitting
from libs import storage
from libs.models import Repo
from libs.proxies import summaries

gpt_model = 'gpt-3.5-turbo'
github_url = os.environ['GITHUB_URL']
enriched_page_content_fmt = '#{file_path}\n#{summary}\n{content}'


def load_repo(url: str, temp_path: str, branch='main') -> Repo:
    git_loader = GitLoader(clone_url=url, repo_path=temp_path, branch=branch)
    repo = Repo(
        name=url.rsplit('/', maxsplit=1)[-1],
        branch=branch,
        url=url,
        documents=git_loader.load(),
        tree=str(Tree(temp_path, absolute=False))
    )
    return repo


from langchain.retrievers.document_compressors import FlashrankRerank

compressor = FlashrankRerank(top_n=7)


def get_conversation_chain(vectorstore):
    langchain.verbose = False
    llm = ChatOpenAI(model=gpt_model, temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Define your system message template
    general_system_template = """You are a clever programming assistant. You answer questions
    in a concise, informative, friendly yet imperative tone.

    The user will ask a question about their codebase, and you will answer it. Using the contextual
    code fragments and documentation provided. You will not make any assumptions about the codebase
    beyond what is presented to you as context.

    When the user asks their question, you will answer it by using the provided code fragments.
    Answer the question using the code below. Explain your reasoning in simple steps. Be assertive!
    
    ----------------
    
    {context}
    
    """
    # Define your user message template
    general_user_template = "Question:```{question}```"

    # Create message prompt templates from your message templates
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        general_system_template
    )
    user_message_prompt = HumanMessagePromptTemplate.from_template(
        general_user_template
    )

    # Create a chat prompt template from your message prompt templates
    qa_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, user_message_prompt]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever()
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return conversation_chain


class Request(BaseModel):
    query: str

    class Config:
        schema_extra = {
            "examples": [
                {
                    "query": "How do I set up a miner?"
                }
            ]
        }


app = FastAPI()


def collection_exists():
    db = storage.vector_db
    return db._collection.count() > 1


class MissingReadme(Exception):
    pass


extra_readme_append_fmt = """
***
Document path: {file_path}
Document contents:
{page_content}
***
"""


def merge_readmes(readme, other_readmes):
    enriched_page_content = readme.page_content
    referenced_files = {}
    for line in readme.page_content.splitlines():
        for other_readme in other_readmes:
            if other_readme.metadata['file_path'] in line:
                referenced_files[other_readme.metadata['file_path']] = other_readme

    if referenced_files:
        for file_path, other_readme in referenced_files.items():
            enriched_page_content += extra_readme_append_fmt.format(
                file_path=file_path,
                page_content=other_readme.page_content
            )

    readme.metadata['enriched_page_content'] = enriched_page_content

    return readme


def find_readme(documents: List[Document]) -> Document:
    other_readmes = []
    root_readme = None
    for index, document in enumerate(documents):

        if document.metadata['file_path'].lower() == 'readme.md':
            # This must be the root repo readme file
            root_readme = document
        elif document.metadata['file_name'].lower().endswith('.md'):
            # These are some other readme files, probably relevant still
            other_readmes.append(document)

    if root_readme:
        if other_readmes:
            root_readme = merge_readmes(root_readme, other_readmes)

    else:
        raise MissingReadme('no main readme file found')

    return root_readme


@app.post("/query/")
async def query(query: Request):
    client = httpx.AsyncClient()
    try:
        if not collection_exists():  # db does not exist
            with tempfile.TemporaryDirectory() as local_path:
                repo = load_repo(github_url, local_path, branch="main")
                try:
                    readme = find_readme(repo.documents)
                except MissingReadme:
                    # todo: what do if repo is missing a main readme file?
                    raise
                else:
                    logger.debug(f'Found root readme file {readme.metadata["file_path"]}')
                    repo_summary_task = summaries.SummarizeRepo(
                        content=readme.metadata['enriched_page_content'],
                        repo_name=repo.name,
                        tree=repo.tree,
                    )
                    repo_summary = await summaries.produce(repo_summary_task, client)
                    logger.debug(f'Summarized repo summary task: {repo_summary}')

                    repo.metadata['summary'] = repo_summary
                    chunks = await splitting.split_documents(
                        documents=repo.documents,
                        repo=repo,
                        client=client
                    )

                    db = storage.vector_db
                    logger.debug(f'Found {len(chunks)} chunks')
                    db.add_documents(chunks)

        vectorstore = storage.vector_db
        convo_chain = get_conversation_chain(vectorstore)
        response = convo_chain({"question": query.query})
        return response
    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")
