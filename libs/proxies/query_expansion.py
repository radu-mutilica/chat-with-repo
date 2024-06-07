from libs.models import ProxyLLMTask, Model
from libs.proxies.providers import corcel

expansion = Model(name='cortext-ultra', provider=corcel, endpoint='text/cortext/chat')


def split_response(_, text: str):
    return text.splitlines()


class QueryExpander(ProxyLLMTask):
    model = expansion

    extra_settings = {
        "temperature": 0.0001,
        "stream": False,
        "top_p": 1,
        "max_tokens": 4096
    }

    system_prompt = """You are an expert at converting user questions into database queries. \
    You have access to a database of tutorial videos about a software library for building \
    LLM-powered applications. \
    
    Perform query expansion. If there are multiple common ways of phrasing a user question \
    or common synonyms for key words in the question, make sure to return multiple versions \
    of the query with the different phrasings.
    
    If there are acronyms or words you are not familiar with, do not try to rephrase them.
    
    *** Example question: ***
    What is the latest subnet release?
    
    *** Example answer: ***
    What is the most recent subnet update?
    What is the current subnet version release?
    What is the latest release of the subnet?
    What is the latest version of the subnet released?
    
    Important:
     - The question is about a specific github repository, so it is a programming question.
     - Make use of synonyms for key words in the question (example: "update" is a synonym for \
     "release")
    
    Return 4 versions of the question."""
    user_prompt = """
    Question: {question}
    """
    post_processing_func = split_response
