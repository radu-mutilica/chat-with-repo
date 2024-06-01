import httpx

from libs.proxies import config
from libs.models import ProxyResponse, QueryPrompts

_assistant_prefix = """You are an expert programming assistant. You provide concise, informative 
and friendly answers to questions you are given. Your task is to """

timeout = httpx.Timeout(10.0, read=None)


class SummaryTask:
    system_prompt = ''
    user_prompt = ''
    model = config.summaries

    def __init__(self, **kwargs):
        system_prompt = self.system_prompt.format(**kwargs)
        user_prompt = self.user_prompt.format(**kwargs)

        self._prompts = QueryPrompts(system=system_prompt, user=user_prompt)

    @property
    def prompts(self):
        return self._prompts


class SummarizeRepo(SummaryTask):
    system_prompt = _assistant_prefix + """ come up with a description of what you believe a
    github repository does. For context, you will be given the repository's Readme, as well
    as the repository's file structure. Do not make any assumptions, stick to the context and if you
    can't solve the task, say "I don't know".
    """
    user_prompt = """The repository is called '{repo_name}'
    The Readme file contains:
    ```md
    {content}
    ```
    The repository file structure looks like:
    ```sh
    {tree}
    ```

    Describe, based on the Readme, file structure and file names what this code does and how it 
    is structured."""


class SummarizeFile(SummaryTask):
    system_prompt = _assistant_prefix + """come up with a description of what you believe a file 
    containing programming code does. You will be also be given the originating repository's 
    description, as well as the repository's file structure, to aid you with context.
    
    Your job is to produce a short description of what you believe the code in this file achieves.
    Keep it to 1-2 sentences maximum. Use present tense and try to be as concise as possible. 
    
    A good example of a summary is:
    "FastAPI's endpoints declaration for the LLM server"
    """

    user_prompt = """The repository is called '{repo_name}'.
    
    Description:
    {repo_summary}
    
    The file structure looks like:
    ```sh
    {tree}
    ```
    
    The file you have to summarize is: '{file_path}'
    ```{language}
    {content}
    ```
    
    Produce a short 1 sentence summary of the contents of the above file, in context of the rest 
    of the repository.
    """


class SummarizeSnippet(SummaryTask):
    system_prompt = _assistant_prefix + """come up with a description of a short code snippet, from
    a larger code paragraph, contained in a file present in a greater repository. For context in
    summarizing this snippet, you will be given the immediate code context it is present in, the 
    description of the file it is found in, as well as the the general description of the repository
    it is contained in.
    
    A good example of a summary is: 
    "Initialize OpenAI and other custom proxies API credentials" 
    
    or
    
    "Introduction section of the 'vision' repository's README file. Overview of the 
    repository's purpose related to Bittensor and decentralized subnet inference."
    """

    user_prompt = """The repository is called '{repo_name}'.
    Description: {repo_summary}
    
    The file structure looks like:
    ```sh
    {tree}
    ```
    
    Based on this fragment of code from '{file_path}':
    ```{language}
    {context}
    ```
    
    What does the following snippet do?
    ```
    {content}
    ```
    
    Provide a short, condensed 1 sentence summary of what you believe the snippet does.
    """


async def produce(summary_task: SummaryTask, client: httpx.AsyncClient) -> ProxyResponse:
    payload = {
        "model": summary_task.model.name,
        "messages": summary_task.prompts.api_format()
    }

    response = await client.post(
        url=summary_task.model.url,
        json=payload,
        headers=summary_task.model.provider.headers,
        timeout=timeout)

    response.raise_for_status()
    raw_response = response.json()
    return raw_response['choices'][0]['message']['content']
