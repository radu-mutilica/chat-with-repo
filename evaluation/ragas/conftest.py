import pytest

from data_generation import load_test_set

test_set_collection = 'subnet19'


@pytest.fixture(scope='session')
def test_set():
    yield load_test_set(collection_name=test_set_collection)
