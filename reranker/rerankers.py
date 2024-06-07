import logging
import os
import time
from typing import List, Dict

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

crossencoder_top_k = int(os.environ['CROSSENCODER_TOP_K'])
crossencoder_model = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-2-v2',
    max_length=512
)


async def crossencoder(query: str, documents: List[str]) -> List[Dict[str, int|float]]:
    """Use a crossencoder model to compute a document rank.

    Args:
        query: (str) the user's query string to compare against.
        documents: (list) a collection of raw document texts.

    Returns:
        A list of scores (floats).
    """
    start = time.time()

    ranks = crossencoder_model.rank(query, documents, top_k=crossencoder_top_k)

    logger.debug(
        f'Crossencoder took {round(time.time() - start, 2)}s and generated '
        f'following rankings{ranks}')
    logger.debug(
        f'Crossencoder dropped {len(documents) - len(ranks)} docs after applying '
        f'a top_k={crossencoder_top_k}'
    )

    return ranks
