import logging
import os
import pathlib

from dotenv import load_dotenv

from libs.models import Provider, Model

load_dotenv()


log_level = logging.DEBUG

chroma_persist_directory = str(pathlib.Path.cwd() / 'chroma')

openai = Provider(
    name='OpenAI',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    },
    url='https://api.openai.com/v1'
)

corcel = Provider(
    name='Corcel',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['CORCEL_API_KEY']}"
    },
    url='https://api.corcel.io/v1'
)

embeddings = Model(name='text-embedding-3-small', provider=openai, endpoint='embeddings')
summaries = Model(name='gpt-3.5-turbo', provider=openai, endpoint='chat/completions')
