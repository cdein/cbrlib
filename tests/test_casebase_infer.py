from dataclasses import dataclass
import functools
from typing import Optional

from cbrlib import casebase, evaluate
from cbrlib.casebase import ReasoningRequest, ReasoningResponse


@dataclass
class DataObject:
    color: Optional[str] = None
    shape: Optional[str] = None
    pattern: Optional[str] = None
    another_color: Optional[str] = None


lookup = {
    "red": {"red": 1, "orange": 0.8, "yellow": 0.4},
    "orange": {"orange": 1, "red": 0.8, "yellow": 0.8},
}
lookup_evaluator = functools.partial(evaluate.table_lookup, lookup)


mapping = (
    evaluate.WeightedPropertyEvaluatorMapping("color", evaluate.equality, 1),
    evaluate.WeightedPropertyEvaluatorMapping("shape", evaluate.equality, 1),
    evaluate.WeightedPropertyEvaluatorMapping("another_color", lookup_evaluator, 1),
)

dataobject_equality_evaluator = functools.partial(
    evaluate.case_average,
    mapping,
)


def test_evaluate_casebase_equality() -> None:
    request = ReasoningRequest(query=DataObject(color="red"))
    response = casebase.infer(
        [
            DataObject(color="red"),
            DataObject(color="blue"),
            DataObject(color="green"),
        ],
        request,
        dataobject_equality_evaluator,
    )
    assert isinstance(response, ReasoningResponse)
    assert response.total_number_of_hits == 1
    assert response.hits[0].similarity == 1


def test_evaluate_casebase_similarity() -> None:
    request = ReasoningRequest(query=DataObject(another_color="red"))
    response = casebase.infer(
        [
            DataObject(another_color="red"),
            DataObject(another_color="orange"),
            DataObject(another_color="green"),
        ],
        request,
        dataobject_equality_evaluator,
    )
    assert isinstance(response, ReasoningResponse)
    assert response.total_number_of_hits == 2
    assert response.hits[0].similarity == 1
    assert response.hits[1].similarity == 0.8
