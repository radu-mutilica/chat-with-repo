import logging
import os
from typing import List

import langchain
from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from libs import storage

logger = logging.getLogger()
logger.setLevel(os.environ['LOG_LEVEL'])
handler = logging.StreamHandler()
handler.setLevel(os.environ['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Message(BaseModel):
    role: str
    content: str


class Request(BaseModel):
    reranker: str = None
    temperature: float = None
    messages: List[Message]


app = FastAPI()

system_prompt_fmt = """You are a clever programming assistant. You answer questions
    in a concise, informative, friendly yet imperative tone.

    The user will ask a question about their codebase, and you will answer it. Using the contextual
    code fragments and documentation provided. You will not make any assumptions about the codebase
    beyond what is presented to you as context.

    When the user asks their question, you will answer it by using the provided code fragments.
    Answer the question using the code below. Explain your reasoning in simple steps. Be assertive
    and quote code fragments if needed.

    ----------------

    {context}
    """

user_prompt_fmt = "Question: {question}"

reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
compressor = CrossEncoderReranker(model=reranker, top_n=3)


def get_conversation_chain(vectorstore):
    langchain.verbose = False
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.5
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    general_system_template = system_prompt_fmt
    general_user_template = user_prompt_fmt

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        general_system_template
    )
    user_message_prompt = HumanMessagePromptTemplate.from_template(
        general_user_template
    )

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


@app.post("/chat/")
async def chat_with_repo(request: Request):
    try:
        conversation_chain = get_conversation_chain(storage.vector_db)
        response = conversation_chain({"question": request.messages[0].content})
        return response
    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")
