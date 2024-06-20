import datetime
import os
import pathlib

import chromadb
from chromadb import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from deepeval.dataset import EvaluationDataset

test_set_json_path = pathlib.Path(__file__).parent.resolve() / 'data' / '{collection}'


def find_last_dataset(path: str):
    datasets = []
    for f in os.listdir(path):
        if f.endswith('.json'):
            datasets.append(f)

    # 20240620_101401
    datasets = sorted(datasets, key=lambda x: parse_ts(x.split('.')[0]), reverse=True)

    return datasets[0]


def parse_ts(timestamp: str):
    return datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S").timestamp()


def _load_data_set(path):
    dataset = EvaluationDataset()
    last_dataset_json = find_last_dataset(path)
    full_dataset_json_path = os.path.join(path, last_dataset_json)

    dataset.add_test_cases_from_json_file(
        file_path=full_dataset_json_path,
        input_key_name='input',
        actual_output_key_name='actual_output',
        expected_output_key_name='expected_output',
        context_key_name='context',
        retrieval_context_key_name='retrieval_context'
    )
    return dataset


def load_test_set(collection_name: str):
    collection_test_set_json_path = str(test_set_json_path).format(collection=collection_name)
    print('Looking for dataset files in {}'.format(collection_test_set_json_path))
    try:
        candidates = os.listdir(collection_test_set_json_path)
    except FileNotFoundError:
        print('Path does not exist, creating...')
        os.makedirs(collection_test_set_json_path)
    else:
        print('Found {} candidates'.format(len(candidates)))
        print(candidates)

    if os.path.isdir(collection_test_set_json_path):
        print('Loading test set from json file: {}'.format(collection_test_set_json_path))
        try:
            return _load_data_set(collection_test_set_json_path)
        except IndexError:
            print('No recognized datasets for {}'.format(collection_name))
            return synthesize(collection_name)
    else:
        print('Synthesizing new test set from collection {}'.format(collection_name))
        return synthesize(collection_name)


def get_db(collection):
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


def synthesize(collection_name: str):
    contexts = {}
    collection = get_db(collection_name)
    data = collection.get(include=['documents', 'metadatas'])
    for doc, meta in zip(data['documents'], data['metadatas']):
        try:
            contexts[meta['source']].append(doc)
        except KeyError:
            contexts[meta['source']] = [doc]

    dataset = EvaluationDataset()
    dataset.generate_goldens(
        contexts=list(contexts.values())[:3]
    )
    dataset.save_as(
        file_type='json',
        directory=str(test_set_json_path).format(collection=collection_name)
    )
    return dataset
