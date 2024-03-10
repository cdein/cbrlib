from dataclasses import dataclass
import dataclasses
from enum import Enum, auto
from typing import Any, Generic, Iterable, Optional, TypeVar


C = TypeVar("C")


@dataclass(slots=True)
class FacetValue:
    value: Any
    count: int = 0


@dataclass(slots=True, frozen=True)
class Facet:
    name: str
    values: list[FacetValue]
    entropy: Optional[float] = dataclasses.field(default=0)


class FacetOrderCriteria(Enum):
    COUNT = auto()
    VALUE = auto()


@dataclass(slots=True, frozen=True)
class FacetConfig:
    name: str
    max_count: Optional[int] = dataclasses.field(default=5)
    order_by: Optional[FacetOrderCriteria] = dataclasses.field(default=FacetOrderCriteria.COUNT)


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
