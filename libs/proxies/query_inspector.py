from libs.models import ProxyLLMTask, Model
from libs.proxies.providers import corcel

query_rephrase = Model(name='cortext-ultra', provider=corcel, endpoint='text/cortext/chat')
chat_message_fmt = '{role}: {content}'


def strip_new_query(_, text: str) -> str:
    """Helper func to remove prefix from LLM response"""
    assert 'Rephrased: ' in text
    return text[10:]


def format_chat_history(_, kwargs) -> str:
    """Helper func to string format a collection of chat history messages"""
    raw_chat_history = kwargs.pop('chat_history')

    kwargs['chat_history'] = '\n'.join(
        chat_message_fmt.format(role=message.role, content=message.content.raw)
        for message in raw_chat_history
    )

    return kwargs


class RephraseGivenHistory(ProxyLLMTask):
    extra_settings = {
        "stream": False,
        "top_p": 1,
        "temperature": 0.1,
        "max_tokens": 512
    }
    model = query_rephrase
    system_prompt = """System: You are an AI assistant that rephrases user queries based on \
    conversation context. Analyze the historical chat and current query, then generate a rephrased \
    query considering the following:

        1. Identify the main topic or intent of the current query.
        2. Find relevant context from the historical chat to clarify or expand the query.
        3. Incorporate the context into the rephrased query to make it more specific or informative.
        4. Maintain the core meaning of the original query while enhancing it with context.
        5. If no relevant context exists, rephrase the query for clarity or grammatical correctness.

    Example:
    
        Chat history:
            User: I'm interested in learning about the history of Rome.
            Assistant: Rome has a rich history dating back to 753 BCE, growing from a small town \
            to a large empire. Are there any specific aspects you'd like to know more about?
        
        Latest query: Tell me about the early years.
        
        Rephrased: Tell me about the Rome's history from 753 BCE.
    
    Format your response as:
    
    Rephrased: [Your rephrased query here]"""

    user_prompt = """Given the following chat history:
    {chat_history}
    
    Rephrase this latest query: {query}
    """
    pre_processing_func = format_chat_history
    post_processing_func = strip_new_query
