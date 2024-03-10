from dataclasses import dataclass
import functools
from typing import Optional

from cbrlib import casebase, evaluate
from cbrlib.casebase import ReasoningRequest, ReasoningResponse
from cbrlib.types import FacetConfig


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


def test_evaluate_casebase_facets() -> None:
    request = ReasoningRequest(
        query=DataObject(color="red"),
        threshold=0,
        facets=(
            FacetConfig("color", 4),
            FacetConfig("irrelevant", 4),
        ),
    )
    response = casebase.infer(
        [
            DataObject(color="red"),
            DataObject(color="red"),
            DataObject(color="red"),
            DataObject(color="red"),
            DataObject(color="red"),
            DataObject(color="green"),
            DataObject(color="green"),
            DataObject(color="green"),
            DataObject(color="green"),
            DataObject(color="yellow"),
            DataObject(color="yellow"),
            DataObject(color="yellow"),
            DataObject(color="orange"),
            DataObject(color="orange"),
            DataObject(color="blue"),
            DataObject(color="purple"),
            DataObject(color="magenta"),
            DataObject(),  # Allow also empty objects
            DataObject(color=None),  # or missing property value
        ],
        request,
        dataobject_equality_evaluator,
    )
    assert isinstance(response, ReasoningResponse)
    assert response.facets is not None
    assert len(response.facets) == 1
    assert len(response.facets[0].values) == 4
    assert response.facets[0].values[0].value == "red"
    assert response.facets[0].values[0].count == 5
    assert response.facets[0].values[1].value == "green"
    assert response.facets[0].values[1].count == 4
    assert response.facets[0].values[2].value == "yellow"
    assert response.facets[0].values[2].count == 3
    assert response.facets[0].values[3].value == "orange"
    assert response.facets[0].values[3].count == 2
