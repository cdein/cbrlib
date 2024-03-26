import dataclasses
import functools
from typing import Optional

from cbrlib import (
    Evaluator,
    FunctionCalculationParameter,
    NumericEvaluationOptions,
    ReasoningRequest,
    WeightedPropertyEvaluatorMapping,
)
from cbrlib import casebase, evaluate


@dataclasses.dataclass(slots=True, frozen=True)
class DataObject:
    color: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[int] = None


color_lookup = {
    "red": {"orange": 0.8, "yellow": 0.4},
    "orange": {"red": 0.8, "yellow": 0.8},
}
color_evaluator = functools.partial(evaluate.table_lookup, color_lookup)


def create_size_evaluator() -> Evaluator:
    options = NumericEvaluationOptions(
        min_=0,
        max_=100,
        if_less=FunctionCalculationParameter(tolerance=1.0),
        if_more=FunctionCalculationParameter(tolerance=1.0),
    )
    return functools.partial(evaluate.numeric, options)


def dataobject_evaluator() -> Evaluator:
    mappings = (
        WeightedPropertyEvaluatorMapping("color", color_evaluator, 2),
        WeightedPropertyEvaluatorMapping("size", create_size_evaluator(), 1),
        WeightedPropertyEvaluatorMapping("shape", evaluate.equality, 1),
    )
    return functools.partial(evaluate.case_average, mappings)


def main() -> None:
    data = [
        DataObject(color="red", shape="triangle", size=20),
        DataObject(color="orange", shape="circle", size=70),
        DataObject(color="green", shape="square", size=50),
    ]
    print("-" * 80)
    print(
        casebase.infer(
            data,
            ReasoningRequest(
                query=DataObject(color="red"),
            ),
            dataobject_evaluator(),
        )
    )
    print("-" * 80)
    print(
        casebase.infer(
            data,
            ReasoningRequest(
                query=DataObject(size=50),
            ),
            dataobject_evaluator(),
        )
    )


if __name__ == "__main__":
    main()
