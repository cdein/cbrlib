from __future__ import annotations

import dataclasses
import functools
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

C = TypeVar("C")


@dataclass(slots=True)
class FacetValue:
    value: Any
    count: int = 0
    importance: Optional[float] = 0


@dataclass(slots=True, frozen=True)
class Facet:
    name: str
    values: list[FacetValue]
    entropy: Optional[float] = dataclasses.field(default=0)


class FacetValueProperty(Enum):
    COUNT = "count"
    VALUE = "value"
    IMPORTANCE = "importance"


class FacetValueOrder(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(slots=True, frozen=True)
class FacetValueOrderCriteria:
    property: Optional[FacetValueProperty] = dataclasses.field(default=FacetValueProperty.IMPORTANCE)
    order: Optional[FacetValueOrder] = dataclasses.field(default=FacetValueOrder.DESCENDING)

    @functools.cache
    def as_key_function(self) -> Callable[[FacetValue], Any]:
        if self.property == FacetValueProperty.COUNT:
            return lambda fv: fv.count
        if self.property == FacetValueProperty.VALUE:
            return lambda fv: fv.value
        if self.property == FacetValueProperty.IMPORTANCE:
            return lambda fv: fv.importance

    @functools.cache
    def is_reverse(self):
        return self.order == FacetValueOrder.DESCENDING

    @staticmethod
    @functools.cache
    def importance() -> FacetValueOrderCriteria:
        return FacetValueOrderCriteria(FacetValueProperty.IMPORTANCE, FacetValueOrder.DESCENDING)

    @staticmethod
    @functools.cache
    def value() -> FacetValueOrderCriteria:
        return FacetValueOrderCriteria(FacetValueProperty.VALUE, FacetValueOrder.ASCENDING)

    @staticmethod
    @functools.cache
    def count() -> FacetValueOrderCriteria:
        return FacetValueOrderCriteria(FacetValueProperty.COUNT, FacetValueOrder.DESCENDING)


@dataclass(slots=True, frozen=True)
class FacetConfig:
    name: str
    max_count: Optional[int] = dataclasses.field(default=5)
    order_by: Optional[FacetValueOrderCriteria] = dataclasses.field(default_factory=FacetValueOrderCriteria.importance)


@dataclass(slots=True, frozen=True)
class Result(Generic[C]):
    similarity: float
    case: C


@dataclass(slots=True, frozen=True)
class ReasoningRequest(Generic[C]):
    query: C

    offset: Optional[int] = dataclasses.field(default=0)
    limit: Optional[int] = dataclasses.field(default=10)

    threshold: Optional[float] = dataclasses.field(default=0.1)

    facets: Optional[Iterable[FacetConfig]] = None


@dataclass(slots=True, frozen=True)
class ReasoningResponse(Generic[C]):
    total_number_of_hits: int
    hits: Iterable[Result]
    facets: Optional[Iterable[Facet]]


Evaluator = Callable[[Any, Any], float]


WeightedPropertyEvaluatorMapping = namedtuple(
    "WeightedPropertyEvaluatorMapping",
    {"property_name", "evaluator", "weight"},
)


PropertyEvaluatorMapping = namedtuple(
    "PropertyEvaluatorMapping",
    {"property_name", "evaluator"},
)


def _interpolate_polynom(stretched_distance: float, linearity: float) -> float:
    if linearity == 0:
        return 0.0
    elif linearity == 1:
        return 1 - stretched_distance
    return pow(1 - stretched_distance, 1 / linearity)


def _interpolate_root(stretched_distance: float, linearity: float) -> float:
    if linearity == 0:
        return 1.0
    elif linearity == 1:
        return 1 - stretched_distance
    return pow(1 - stretched_distance, linearity)


def _interpolate_sigmoid(stretched_distance: float, linearity: float) -> float:
    if linearity == 1:
        return 1 - stretched_distance
    if stretched_distance < 0.5:
        if linearity == 0:
            return 1.0
        return 1 - pow(2 * stretched_distance, 1 / linearity) / 2
    if linearity == 0:
        return 0.0
    return pow(2 - 2 * stretched_distance, 1 / linearity) / 2


class NumericInterpolation(Enum):
    POLYNOM = "polynomial"
    SIGMOID = "sigmoid"
    ROOT = "root"


_interpolations = {
    NumericInterpolation.POLYNOM: _interpolate_polynom,
    NumericInterpolation.ROOT: _interpolate_root,
    NumericInterpolation.SIGMOID: _interpolate_sigmoid,
}


@dataclass(slots=True, frozen=True)
class FunctionCalculationParameter:
    equal: float = dataclasses.field(default=0.0)
    tolerance: float = dataclasses.field(default=0.5)
    linearity: float = dataclasses.field(default=1.0)
    interpolation: NumericInterpolation = dataclasses.field(default=NumericInterpolation.POLYNOM)

    @functools.cache
    def get_interpolation(self):
        return _interpolations[self.interpolation]

    @staticmethod
    @functools.cache
    def default() -> "FunctionCalculationParameter":
        return FunctionCalculationParameter()


@dataclass(slots=True, frozen=True)
class NumericEvaluationOptions:
    min_: float
    max_: float

    origin: float = dataclasses.field(default=0)
    use_origin: bool = dataclasses.field(default=False)

    cyclic: bool = dataclasses.field(default=False)

    if_less: FunctionCalculationParameter = dataclasses.field(default=FunctionCalculationParameter.default())
    if_more: FunctionCalculationParameter = dataclasses.field(default=FunctionCalculationParameter.default())

    def __post_init__(self) -> None:
        min_ = self.min_
        max_ = self.max_
        if max_ < min_:
            max_, min_ = min_, max_
        object.__setattr__(self, "min_", min_)
        object.__setattr__(self, "max_", max_)

    @property
    @functools.cache
    def max_distance(self) -> float:
        return self.max_ - self.min_
