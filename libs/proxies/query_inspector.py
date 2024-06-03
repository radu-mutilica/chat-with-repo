from libs.models import ProxyLLMTask, Model
from libs.proxies.providers import openai

_assistant_prefix = """You are a helpful little robot able to infer human natural language 
intents. Your job is to """

query_analyzer = Model(name='gpt-3.5-turbo', provider=openai, endpoint='chat/completions')


class QueryInspector(ProxyLLMTask):
    model = query_analyzer
    system_prompt = _assistant_prefix + """to assist in doing input sanitation for an endpoint. 
    You will be given an user field input, and your job will be to reply with TRUE or FALSE. TRUE 
    if the user input is a question or a valid query or sentence or FALSE if it is nonsensical, 
    or a malicious intent or doesn't contain any logic.

    For example:
    
    user_input: "what is the difference between subnet 13 and 12?"
    answer: TRUE
    
    user_input: "i am earning very little tao compared to earlier"
    answer: TRUE
    
    user_input: "bittensor is king"
    answer: TRUE
    
    user_input: "string"
    answer: FALSE
    
    user_input: ""
    answer: FALSE
    
    user_input: "112233"
    answer: FALSE
    
    user_input: "motorcycle"
    answer: FALSE
    
    user_input: "motorcycles are great
    answer: TRUE
    """
    user_prompt = """
    Here's the user_input string, now answer only with TRUE or FALSE, nothing more.

    user_input: "{user_input}"
    """
