import logging
import os
import pathlib
import tempfile

import langchain
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import GitLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

chroma_local_path = pathlib.Path.cwd() / 'chroma'
gpt_model = 'gpt-3.5-turbo'
github_url = os.environ['GITHUB_URL']


def load_github_repo(github_url, local_path, repo_branch):
    loader = GitLoader(
        clone_url=github_url,
        repo_path=local_path,
        branch=repo_branch,
    )
    docs = loader.load()
    return docs


def split_documents(documents_list):
    split_documents_list = []

    for doc in documents_list:
        try:
            ext = os.path.splitext(doc.metadata["source"])[1]
            lang = get_language_from_extension(ext)
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=512, chunk_overlap=0
            )
            split_docs = splitter.create_documents([doc.page_content])
            for split_doc in split_docs:
                split_doc.metadata.update(
                    doc.metadata
                )  # Copy metadata from original doc
                split_documents_list.append(
                    split_doc
                )  # Store split documents in a list

        except Exception:
            logger.exception(
                f"Error splitting document: {doc.metadata['source']}"
            )
    return split_documents_list


def get_language_from_extension(ext):
    # Simplified mapping from file extension to LangChain Language enum
    ext_to_lang = {
        ".cpp": Language.CPP,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.JS,
        ".tsx": Language.JS,
        ".php": Language.PHP,
        ".proto": Language.PROTO,
        ".py": Language.PYTHON,
        ".rst": Language.RST,
        ".rb": Language.RUBY,
        ".rs": Language.RUST,
        ".scala": Language.SCALA,
        ".swift": Language.SWIFT,
        ".md": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".sol": Language.SOL,
        ".css": Language.HTML,
        ".txt": Language.MARKDOWN,
        ".json": Language.MARKDOWN,
    }
    return ext_to_lang.get(ext, Language.MARKDOWN)


def create_vectorstore(chunks):
    db = Chroma(
        embedding_function=OpenAIEmbeddings(disallowed_special=()),
        persist_directory=str(chroma_local_path)
    )
    db.add_documents(chunks)
    return db


def load_vectorstore():
    db = Chroma(
        embedding_function=OpenAIEmbeddings(disallowed_special=()),
        persist_directory=str(chroma_local_path)
    )
    return db


def get_conversation_chain(vectorstore, gpt_model="gpt-3.5-turbo"):
    langchain.verbose = False
    llm = ChatOpenAI(model=gpt_model, temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Define your system message template
    general_system_template = """You are a superintelligent AI that answers questions about codebases.
    You are:
    - helpful & friendly
    - good at answering complex questions in simple language
    - an expert in all programming languages
    - able to infer the intent of the user's question

    The user will ask a question about their codebase, and you will answer it.

    When the user asks their question, you will answer it by searching the codebase for the answer.
    Answer the question using the code file(s) below:
    ----------------
        {context}"""
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

    print("system prompt", system_message_prompt)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return conversation_chain


class Query(BaseModel):
    user_input: str


app = FastAPI()


def collection_exists():
    client = Chroma(
        embedding_function=OpenAIEmbeddings(disallowed_special=()),
        persist_directory=str(chroma_local_path)
    )

    return client._collection.count() > 1


@app.post("/query/")
async def query(query: Query):
    try:
        if not collection_exists():  # db does not exist
            with tempfile.TemporaryDirectory() as local_path:
                docs = load_github_repo(
                    github_url, local_path, repo_branch="main"
                )
                chunks = split_documents(docs)
                db = create_vectorstore(chunks)
                db.persist()

        vectorstore = load_vectorstore()
        convo_chain = get_conversation_chain(vectorstore, gpt_model=gpt_model)
        response = convo_chain({"question": query.user_input})
        return response
    except Exception:
        logger.exception('Failed to fulfill request because:')
        raise HTTPException(status_code=500, detail="An error occurred while processing the query")
