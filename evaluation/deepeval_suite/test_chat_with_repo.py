import logging

import deepeval
import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from . import synthesis

test_set_collection = 'myself'
test_cases = synthesis.load_test_set(collection_name=test_set_collection)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "test_case",
    test_cases
)
@pytest.mark.asyncio
async def test_chat_with_repo(test_case: LLMTestCase):

    # todo: this is the response from the llm
    test_case.actual_output = 'Hi'

    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")


if __name__ == '__main__':
    pass
