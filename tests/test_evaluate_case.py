from dataclasses import dataclass
import functools
import math
from typing import Optional

from cbrlib import evaluate
import pytest


@dataclass
class DataObject:
    color: Optional[str] = None
    shape: Optional[str] = None
    pattern: Optional[str] = None
    another_color: Optional[str] = None


LOOKUP = {
    "red": {"red": 1, "orange": 0.8, "yellow": 0.4},
    "orange": {"orange": 1, "red": 0.8, "yellow": 0.8},
}
LOOKUP_EVALUATOR = functools.partial(evaluate.table_lookup, LOOKUP)


mapping1 = (
    evaluate.WeightedPropertyEvaluatorMapping("color", evaluate.equality, 1),
    evaluate.WeightedPropertyEvaluatorMapping("shape", evaluate.equality, 1),
)


mapping2 = (
    evaluate.WeightedPropertyEvaluatorMapping("color", evaluate.equality, 1),
    evaluate.WeightedPropertyEvaluatorMapping("shape", evaluate.equality, 2),
)

mapping3 = (
    evaluate.PropertyEvaluatorMapping("color", evaluate.equality),
    evaluate.PropertyEvaluatorMapping("shape", evaluate.equality),
    evaluate.PropertyEvaluatorMapping("pattern", evaluate.equality),
)

mapping4 = (
    evaluate.PropertyEvaluatorMapping("color", LOOKUP_EVALUATOR),
    evaluate.PropertyEvaluatorMapping("shape", evaluate.equality),
    evaluate.PropertyEvaluatorMapping("pattern", evaluate.equality),
    evaluate.PropertyEvaluatorMapping("another_color", LOOKUP_EVALUATOR),
)


def test_case_average() -> None:
    assert (
        evaluate.case_average(
            mapping1,
            DataObject(color="red", shape="triangle"),
            DataObject(color="red", shape="square"),
        )
        == 0.5
    )
    assert (
        evaluate.case_average(
            mapping1,
            DataObject(color="red", shape="triangle"),
            DataObject(color="red", shape="triangle"),
        )
        == 1
    )
    assert (
        evaluate.case_average(
            mapping1,
            DataObject(color="red"),
            DataObject(color="red"),
        )
        == 1
    )
    assert evaluate.case_average(
        mapping2,
        DataObject(color="red", shape="triangle"),
        DataObject(color="red"),
    ) == pytest.approx(1 / 3)


def test_case_average_missing_properties() -> None:
    assert (
        evaluate.case_average(
            mapping2,
            DataObject(color="red"),
            DataObject(),
        )
        == 0
    )
    assert (
        evaluate.case_average(
            mapping2,
            DataObject(),
            DataObject(color="red"),
        )
        == 0
    )


def test_case_median() -> None:
    assert (
        evaluate.case_median(
            mapping3,
            DataObject(color="red", shape="triangle"),
            DataObject(color="red", shape="square"),
        )
        == 0.5
    )
    assert (
        evaluate.case_median(
            mapping3,
            DataObject(color="red", shape="triangle", pattern="dashed"),
            DataObject(color="red", shape="square", pattern="dashed"),
        )
        == 1
    )
    assert (
        evaluate.case_median(
            mapping3,
            DataObject(color="red", shape="triangle", pattern="dashed"),
            DataObject(color="red", shape="square", pattern="dotted"),
        )
        == 0
    )


def test_case_median_missing_properties() -> None:
    assert (
        evaluate.case_median(
            mapping3,
            DataObject(),
            DataObject(color="red", shape="square", pattern="dotted"),
        )
        == 0
    )
    assert (
        evaluate.case_median(
            mapping3,
            DataObject(color="red", shape="square", pattern="dotted"),
            DataObject(),
        )
        == 0
    )


def test_case_min() -> None:
    assert (
        evaluate.case_min(
            mapping4,
            DataObject(color="red", shape="triangle", pattern="dashed"),
            DataObject(color="orange", shape="triangle", pattern="dashed"),
        )
        == 0.8
    )
    assert (
        evaluate.case_min(
            mapping4,
            DataObject(
                color="red", shape="triangle", pattern="dashed", another_color="red"
            ),
            DataObject(
                color="orange",
                shape="triangle",
                pattern="dashed",
                another_color="yellow",
            ),
        )
        == 0.4
    )


def test_case_min_missing_properties() -> None:
    assert (
        evaluate.case_min(
            mapping4,
            DataObject(),
            DataObject(color="orange", shape="triangle", pattern="dashed"),
        )
        == 0
    )
    assert (
        evaluate.case_min(
            mapping4,
            DataObject(color="orange", shape="triangle", pattern="dashed"),
            DataObject(),
        )
        == 0
    )


def test_case_max() -> None:
    assert (
        evaluate.case_max(
            mapping4,
            DataObject(color="red", shape="triangle", pattern="dashed"),
            DataObject(color="orange", shape="triangle", pattern="dashed"),
        )
        == 1
    )
    assert (
        evaluate.case_max(
            mapping4,
            DataObject(color="red", another_color="red"),
            DataObject(
                color="orange",
                another_color="yellow",
            ),
        )
        == 0.8
    )


def test_case_max_missing_properties() -> None:
    assert (
        evaluate.case_max(
            mapping4,
            DataObject(),
            DataObject(color="orange", shape="triangle", pattern="dashed"),
        )
        == 0
    )
    assert (
        evaluate.case_max(
            mapping4,
            DataObject(color="orange", shape="triangle", pattern="dashed"),
            DataObject(),
        )
        == 0
    )


def test_case_euclidean() -> None:
    assert evaluate.case_euclidean(
        mapping4,
        DataObject(color="red", shape="triangle"),
        DataObject(color="orange", shape="triangle"),
    ) == math.sqrt((0.8**2 + 1**2))


def test_case_euclidean_missing_properties() -> None:
    assert (
        evaluate.case_euclidean(
            mapping4,
            DataObject(),
            DataObject(color="orange", shape="triangle"),
        )
        == 0
    )
    assert (
        evaluate.case_euclidean(
            mapping4,
            DataObject(color="orange", shape="triangle"),
            DataObject(),
        )
        == 0
    )
