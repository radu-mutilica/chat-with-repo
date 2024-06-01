import os

from libs.models import Provider

openai = Provider(
    name='OpenAI',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    },
    url='https://api.openai.com/v1'
)