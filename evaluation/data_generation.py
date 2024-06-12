import logging
import os
import pathlib
import time

from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig
from ragas.testset.docstore import Document
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator

from libs.proxies.embeddings import embeddings
from libs.storage import get_db

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

generator_llm_model = 'gpt-3.5-turbo-16k'
critic_llm_model = 'gpt-4'
test_set_json_path = pathlib.Path(__file__).parent.resolve() / 'data' / 'chat-with-repo.json'


def load_test_set(collection_name: str):
    if not os.path.isfile(test_set_json_path):
        collection = get_db(collection_name)
        logger.info(f'Generating test data using collection "{collection_name}"')

        documents = []
        data = collection.get(include=['metadatas', 'documents', 'embeddings'])
        for doc, emb, meta in zip(
                data['documents'], data['embeddings'], data['metadatas']):

            meta['filename'] = meta['file_path']  # needed for Ragas to work
            documents.append(Document(page_content=doc, metadata=meta, embedding=emb))
        logger.info(f'Got {len(documents)} documents from collection"')

        generator_llm = ChatOpenAI(model=generator_llm_model)
        critic_llm = ChatOpenAI(model=critic_llm_model)
        emb_model = OpenAIEmbeddings(model=embeddings.name)

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            emb_model
        )
        logger.info('Prepared generator...')

        test_set = generator.generate_with_langchain_docs(
            documents,
            test_size=1,
            distributions={
                simple: 0.5,
                reasoning: 0.25,
                multi_context: 0.25
            },
            run_config=RunConfig(
                max_workers=1
            ),
            is_async=False,
            raise_exceptions=False

        ).to_dataset()
        logger.info('Finished generating test set!')
        test_set.to_json(test_set_json_path)
        logger.info(f'Saved test set to "{test_set_json_path}"')

        return test_set.to_dict()

    else:
        logger.info('Found test set json file "{test_set_json_path}"')
        return load_dataset(path=str(test_set_json_path.parent))
