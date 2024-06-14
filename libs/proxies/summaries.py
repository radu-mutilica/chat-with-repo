from libs.models import Model, ProxyLLMTask
from libs.proxies.providers import corcel


summaries = Model(name='gpt-4o', provider=corcel, endpoint='text/cortext/chat')


class SummaryTask(ProxyLLMTask):
    extra_settings = {
        "stream": False,
        "top_p": 1,
        "temperature": 0.1,
        "max_tokens": 4096
    }


class SummarizeRepo(SummaryTask):
    model = summaries
    system_prompt = """You are an intelligent repository summarizer assistant. Your task is to 
    provide a concise summary of the key information and purpose of a given repository based on 
    its file tree structure and README file contents.

    Instructions: 
    
    1. Analyze the repository's file tree structure to understand the organization and main 
    components of the repository. 
    
    2. Read and comprehend the content of the repository's README file. 
    
    3. Identify the primary purpose, key features, and main functionalities of the repository and 
    produce a detailed, breakdown report of what each folder contains and each file does. 
    
    5. Ensure that your summary is clear, informative, and accurately represents the repository.
    """

    user_prompt = """The repository is called '{repo_name}'
    The README file contains:
    ```md
    {content}
    ```
    The repository file structure looks like:
    ```sh
    {tree}
    ```
    
    Produce a highly detailed, breakdown report of the repository, its scope, aim, components, what
    each component and each file does.
    """


class SummarizeFile(SummaryTask):
    model = summaries

    system_prompt = """You are an intelligent file summarizer assistant. Your task is to provide 
    an extremely concise summary of the key information in a given file based on the context 
    provided.
    
    Instructions:
    1. Read and understand the context provided. 
    2. Carefully read the file content to be summarized. 
    3. Identify the main topics and critical information covered in the file. 
    4. Provide a compressed summary in 2-3 sentences (maximum 512 characters) that captures the 
    essence of the file's content in relation to the provided documentation. 
    
    Important: 
    Avoid phrases like "The file <x>" or "The <x> file from the <y> repo". Be direct 
    and transactional in your summary to save on word count.
    
    Examples summaries: 
    
    "Specifies patterns for ignoring specific files and directories in version 
    control, such as compiled files, cache directories, logs, and temporary files, among others"
    
    "SQL file which contains commands to drop the \'scores\' table and its schema with fields 
    like id, axon_uid, hotkey, response_time, score, synapse, valid_response, quickest_response, 
    checked_with_server, and timestamp"
    
    "Configuration file sets up configurations for mining and validation processes, including 
    defining paths, network settings, and wallet information through CLI arguments"
    """

    user_prompt = """The repository is '{repo_name}'.
    
    Description:
    {repo_summary}
    
    The file structure:
    ```sh
    {tree}
    ```
    
    The file you have to summarize is: '{file_path}'
    ```{language}
    {content}
    ```
    
    Produce a short summary of the file contents, and use as few words as possible.
    """


class SummarizeSnippet(SummaryTask):
    model = summaries

    system_prompt = """You are an intelligent code summarizer assistant. Your task is to provide 
    an extremely concise summary of the key information in a given code fragment based on the 
    context provided.
    
    Instructions:
    1. Read and understand the context provided. 
    2. Carefully read the code to be summarized. 
    3. Identify the role of the code in relation to the provided documentation. 
    4. Provide a concise summary in 2-3 sentences (maximum 512 characters) that captures the 
    essence of the code.
    5. Ensure that your summary is clear, informative, and accurately represents the repository.
    6. Do not make any assumptions or pursue "what-if" scenarios.
    
    Important: 
    
    Avoid phrases like "The provided code snippet" or "The repository contains." Be 
    direct and transactional in your summary.
    
    Example code summaries:
    
    "Initialize OpenAI and other custom proxies API credentials"
    
    "Introduction section of the 'vision' repository's README file. Overview of the 
    repository's purpose related to Bittensor and decentralized subnet inference."
    """

    user_prompt = """The repository is called '{repo_name}'.
    Description: {repo_summary}
    
    The file structure looks like:
    ```sh
    {tree}
    ```
    
    You have the following file '{file_path}' which has the contains the following:
    {file_summary}
    
    Here is a fragment of code from '{file_path}':
    ```{language}
    {context}
    ```
    
    What does the following code do?
    ```
    {content}
    ```
    
    Provide a short, condensed summary of what you believe the snippet does.
    """
