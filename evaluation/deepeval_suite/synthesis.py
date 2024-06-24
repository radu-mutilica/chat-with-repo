import logging
import os
import pathlib
import time

import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from .utils import RAGChunker
from deepeval.synthesizer import doc_chunker
# Monkey Patching the DocumentChunker with MyDocumentChunker
doc_chunker.DocumentChunker = RAGChunker

test_set_json_path = pathlib.Path(__file__).parent.resolve() / 'data' / '{collection}'

logger = logging.getLogger(__name__)


def load_test_set(collection_name: str) -> EvaluationDataset:
    """Try to load a test set, else synthesize it from the collection"""
    try:
        time.sleep(200)
        dataset = EvaluationDataset()
        dataset.pull(alias=collection_name)
        return dataset
    except Exception:
        logger.exception(f'Failed to load dataset="{collection_name}":')
        dataset = synthesize(collection_name)
        dataset.push(collection_name, overwrite=True)
        return dataset


def get_db(collection):
    """Helper func to load a collection from Chroma. Currently, embedding is hardcoded,
    will look into it later."""
    emb_fn = OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name='text-embedding-3-small'
    )
    vectordb = chromadb.HttpClient(
        host=os.environ['CHROMA_HOST'],
        port=int(os.environ['CHROMA_PORT']),
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    return vectordb.get_collection(
        name=collection,
        embedding_function=emb_fn
    )


def synthesize(collection_name: str) -> EvaluationDataset:
    """Fetch a collection from the vector database and synthesizes it into a dataset.

    Args:
        collection_name (str): The name of the collection to synthesize.

    Returns:
        EvaluationDataset: The new dataset.
    """

    # Attempt to group contexts by source, so they are "related"
    vectors, contexts = get_db(collection_name).get(include=['documents', 'metadatas']), {}
    for doc, meta in zip(vectors['documents'], vectors['metadatas']):
        try:
            contexts[meta['source']].append(doc)
        except KeyError:
            contexts[meta['source']] = [doc]

    dataset = EvaluationDataset()

    # noinspection PyTypeChecker
    dataset.generate_goldens(
        contexts=list(contexts.values()),
        synthesizer=Synthesizer(multithreading=False)
    )

    return dataset
