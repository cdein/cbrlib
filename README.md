[![Python Library cbrlib - main branch](https://github.com/cdein/cbrlib/actions/workflows/library-test-main.yml/badge.svg)](https://github.com/cdein/cbrlib/actions/workflows/library-test-main.yml)
[![PyPi](https://img.shields.io/pypi/v/cbrlib)](https://pypi.org/project/cbrlib/)
# Case Based Reasoning the Pythonic Way

Case Based Reasoning library for Python

This library provides the possibility to integrate with native python. Basic implemetation can use instances of Python classes to calculate similarity between those objects.

You can find a sparse description of case base reasoning at the [english wikipedia page](https://en.wikipedia.org/wiki/Case-based_reasoning). Please also take a look at the [examples](https://github.com/cdein/cbrlib/tree/main/examples).

```python
import dataclasses
import functools
from typing import Optional

from cbrlib impohttps://github.com/cdein/cbrlib/actions/workflows/python-app.ymlrt (
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
    "red": {"red": 1, "orange": 0.8, "yellow": 0.4},
    "orange": {"orange": 1, "red": 0.8, "yellow": 0.8},
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
```

A big thanky you to [myCBR](http://www.mycbr-project.org/) for the example data.
