import logging
import time
from typing import List

from fastapi import FastAPI

from libs.models import RequestData, DocumentRank

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

models = {'crossencoder': rerankers.crossencoder}

app = FastAPI()


@app.post("/rerank")
async def rerank(request: RequestData) -> List[DocumentRank]:
    """Reranker endpoint. Given a query and a list of documents, it will rerank the documents
    according to the similarity to the provided query.

    Args:
        request: (RequestData) the issued request, containing the query and the documents text
        to rerank.

    Returns:
        A list of ranks and corpus ids. A cutoff is also enforced.
    """
    reranker = models[request.model]
    content = request.messages[0].content

    start = time.time()
    ranks = await reranker(
        query=content.query,
        documents=content.documents
    )
    logger.debug(f"New ranks generated in {round(time.time() - start, 2)}s: {ranks}")

    return [DocumentRank.parse_obj(rank) for rank in ranks]
