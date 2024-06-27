import logging

import deepeval
import httpx
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase

import synthesis
from libs.models import Message
from libs.rag import answer_query
from utils import consume_stream

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
test_set_collection = 'myself'


@pytest.mark.parametrize(
    'test_case',
    synthesis.load_test_set(collection_name=test_set_collection)
)
@pytest.mark.asyncio
async def test_chat_with_repo(test_case: LLMTestCase):
    last_message = Message.parse_obj(
        {
            'role': 'user',
            'content': {
                'query': test_case.input,
                'repo': test_set_collection
            }
        }
    )

    rag_response = await answer_query(
        last_message=last_message,
        chat_history=[],
        client=httpx.AsyncClient()
    )
    test_case.retrieval_context = [c.page_content for c in rag_response.context]
    test_case.actual_output = await consume_stream(rag_response.stream)

    hallucination_metric = HallucinationMetric(
        threshold=0.3,
        include_reason=True
    )
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        include_reason=True
    )
    context_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )

    ragas_metric = RagasMetric(threshold=0.5)

    assert_test(test_case, [
        hallucination_metric,
        answer_relevancy_metric,
        context_relevancy_metric,
        faithfulness_metric,
        ragas_metric
    ], run_async=False)


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")
