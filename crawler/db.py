import logging
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)
emb_fn = OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name='text-embedding-3-small'
)


class VectorDBCollection:
    """Context manager to handle collection creation/deletion for ChromaDB"""

    def __init__(self, collection_name):
        logger.info(f'Establishing connection to ChromaDB')
        self._main_collection = collection_name
        self._temp_collection = f'{self._main_collection}.temp'
        self._db_client = chromadb.HttpClient(
            host=os.environ['CHROMA_HOST'], port=int(os.environ['CHROMA_PORT']),
            # todo: check what allow_reset does
            settings=Settings(allow_reset=True, anonymized_telemetry=False))
        logger.info(f'Received heartbeat from database {self._db_client.heartbeat()}')

    def __enter__(self):
        """Do some cleaning before the start of the database operations"""

        # To free up some space, in case of a failed previous run, check for any
        # temp collection present and delete it
        if self._temp_collection in [c.name for c in self._db_client.list_collections()]:
            logger.info(
                f'Found previous {self._temp_collection} collection in database, deleting it')
            self._db_client.delete_collection(name=self._temp_collection)

        # Now prepare a new empty one - this will be used for all the inserts within
        # this client session
        logger.info(f'Creating new {self._temp_collection} collection')
        return self._db_client.create_collection(
            name=self._temp_collection,
            embedding_function=emb_fn)

    def __exit__(self, exc_type, exc_value, traceback):
        """Upon exiting, delete the MAIN collection and replace it with the TEMP one."""
        try:
            self._db_client.delete_collection(name=self._main_collection)
            logger.info(f'Deleted previous MAIN collection {self._temp_collection}')
        except Exception as e:
            # TODO: I may be dumb but I think Chroma wraps any exceptions like ValueError
            # in generic exception. Why?
            logger.critical(f'Failed to delete collection {self._main_collection} because {e}')

        # Only continue if we managed to delete the collection, otherwise the rename
        # will most likely fail
        logger.info(f'Hot-swapping {self._main_collection} with {self._temp_collection}')

        new_collection = self._db_client.get_collection(
            name=self._temp_collection,
            embedding_function=emb_fn
        )
        new_collection.modify(name=self._main_collection)

        logger.info(f'Successfully created new {self._main_collection} '
                    f'with a total vector count of {new_collection.count()}')
