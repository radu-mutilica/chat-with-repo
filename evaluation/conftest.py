import os

from data_generation import load_test_set

import pytest


@pytest.fixture(scope='session')
def test_set():
    yield load_test_set(collection_name=os.environ['CHROMA_COLLECTION'])

