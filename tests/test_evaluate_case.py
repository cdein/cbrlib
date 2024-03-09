from dataclasses import dataclass
from typing import Optional

from cbrlib import evaluate
import pytest


@dataclass
class DataObject:
    color: Optional[str] = None
    shape: Optional[str] = None
    pattern: Optional[str] = None


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
