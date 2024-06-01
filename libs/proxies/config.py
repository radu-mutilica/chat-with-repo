import os

from libs.models import Provider, Model


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
