import logging
import time

from fastapi import FastAPI

from libs.models import RequestData

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# Leave this after logging boilerplate, so we get
# preconfigured logging for the sentence_transformers package
import rerankers

app = FastAPI()

models = {
    'crossencoder': rerankers.crossencoder
}


@app.post("/rerank")
async def rerank(request: RequestData):
    reranker = models[request.model]
    content = request.messages[0].content

    start = time.time()
    ranks = await reranker(
        query=content['query'],
        documents=content['documents']
    )
    logger.debug(f"New ranks generated in {round(time.time() - start, 2)}s: {ranks}")

    return ranks
