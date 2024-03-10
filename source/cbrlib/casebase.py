from dataclasses import dataclass
import dataclasses
from typing import Generic, Iterable, TypeVar

from cbrlib.evaluate import Evaluator


C = TypeVar("C")


@dataclass(slots=True, frozen=True)
class Result(Generic[C]):
    similarity: float
    case: C


@dataclass(slots=True, frozen=True)
class ReasoningRequest(Generic[C]):
    query: C

    offset: int = dataclasses.field(default=0)
    limit: int = dataclasses.field(default=10)

    threshold: float = dataclasses.field(default=0.1)


@dataclass(slots=True, frozen=True)
class ReasoningResponse(Generic[C]):
    total_number_of_hits: int
    hits: Iterable[Result]


def infer(
    casebase: Iterable[C],
    request: ReasoningRequest[C],
    evaluator: Evaluator,
) -> ReasoningResponse[C]:

    threshold = request.threshold
    evaluate_cases = map(
        lambda c: Result(evaluator(request.query, c), c),
        casebase,
    )
    sorted_results = sorted(
        evaluate_cases,
        key=lambda r: r.similarity,
        reverse=True,
    )
    filtered_results = filter(
        lambda r: r.similarity >= threshold,
        sorted_results,
    )
    result_list = list(filtered_results)
    total_number_of_hits = len(result_list)

    offset = request.offset
    limit = request.limit
    return ReasoningResponse(
        total_number_of_hits,
        result_list[offset:offset + limit],  # fmt: skip
    )
