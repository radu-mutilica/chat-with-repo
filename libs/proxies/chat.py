from libs.models import Model, ProxyLLMTask
from libs.proxies.providers import openai

chat = Model(name='gpt-3.5-turbo', provider=openai, endpoint='chat/completions')

_assistant_prefix = """You are an expert programming assistant. You provide concise, informative 
and friendly answers to questions you are given. Your task is to """

contextual_file_fmt = """
File: {path}
Summary: {summary}
"""

contextual_snippet_fmt = """
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
    system_prompt = _assistant_prefix + """answer a question asked about a GitHub repository. 
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
    
    ```"""


def format_context(contextual_chunks):
    context = ''

    files, code_chunks = [], []

    for document in contextual_chunks:
        if document.metadata['source'].endswith('main'):
            files.append(document)
        else:
            code_chunks.append(document)

    for file in files:
        context += contextual_file_fmt.format(
            path=file.metadata['file_path'],
            summary=file.page_content,
        )
        context += '\n'
        context += prompt_separator

    for code_chunk in code_chunks:
        context += contextual_file_fmt.format(
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
