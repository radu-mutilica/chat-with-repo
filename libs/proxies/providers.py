import os

from libs.models import Provider

hf_reranker = Provider(
    name='Huggingface',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['HF_API_KEY']}"
    },
    url=os.environ['HF_RERANKER_API']
)

hf_embeddings = Provider(
    name='Huggingface',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['HF_API_KEY']}"
    },
    url=os.environ['HF_EMBEDDINGS_API']
)

openai = Provider(
    name='OpenAI',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    },
    url='https://api.openai.com/v1'
)

corcel = Provider(
    name='corcel-vision',
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['CORCEL_API_KEY']}"
    },
    url='https://api.corcel.io/v1'
)
