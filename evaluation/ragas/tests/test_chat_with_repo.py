from ragas import evaluate
from ragas.metrics import context_precision


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


async def test_chat_with_repo(test_set):
    result = evaluate(
        test_set['train'],
        metrics=[context_precision],
        in_ci=False,
    )
    assert result["context_precision"] >= 0.95
