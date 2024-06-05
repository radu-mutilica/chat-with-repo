from typing import List

from langchain_core.documents import Document

from libs.models import Model, ProxyLLMTask
from libs.proxies.providers import corcel_vision

chat = Model(name='llama-3', provider=corcel_vision, endpoint='text/vision/chat')

_assistant_prefix = """You are an expert programming assistant. You provide concise, informative 
and friendly answers to questions you are given. Your task is to """

contextual_file_fmt = """
File: {path}
Summary: {summary}
"""

contextual_code_fmt = """
File: {path}
Code: {code}
Summary: {summary}
"""

code_fmt = """```{language}
{raw_code}
"""
prompt_separator = "-" * 10


class ChatWithRepo(ProxyLLMTask):
    model = chat
    system_prompt = _assistant_prefix + """answer a question about a GitHub repository. 
    Using the provided contextual code fragments and documentation provided. You will not make 
    any assumptions about the codebase beyond what is presented to you as context.

    Answer the question using code and documentation below. Explain your reasoning in simple steps. 
    Be assertive and quote code fragments if needed."""

    user_prompt = """
    Here's some relevant documentation:
    
    {context}
    
    ---
    
    Question:
    
    {question}
    """


def format_context(contextual_chunks: List[Document]) -> str:
    """Format the context template to pass to the final llm prompt

    Args:
        contextual_chunks: (List[Document]) a list of contextually related documents.

    Returns:
        str: the context.
    """
    context = ''

    entire_files, isolated_code_chunks = [], []

    for document in contextual_chunks:
        if document.metadata['source'].endswith('main'):
            entire_files.append(document)
        else:
            isolated_code_chunks.append(document)

    for file in entire_files:
        context += contextual_file_fmt.format(
            path=file.metadata['file_path'],
            summary=file.page_content,
        )
        context += '\n'
        context += prompt_separator

    for code_chunk in isolated_code_chunks:
        context += contextual_code_fmt.format(
            path=code_chunk.metadata['file_path'],
            summary=code_chunk.page_content,
            code=code_fmt.format(
                language=code_chunk.metadata['language'],
                raw_code=code_chunk.metadata['original_page_content']
            )
        )
        context += '\n'
        context += prompt_separator

    return context
