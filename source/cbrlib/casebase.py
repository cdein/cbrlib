from typing import Any, Callable, Iterable, Optional

from cbrlib.evaluate import Evaluator
from cbrlib.facetting import FacetCollectingIterator
from cbrlib.types import C, Facet, ReasoningRequest, ReasoningResponse, Result


def _make_relevant_facets(
    query: Any, facets: Iterable[Facet], getvalue: Callable[[Any, str, Optional[Any]], Any]
) -> list[Facet]:
    facet_names = set(f.name for f in facets)
    return list(
        filter(
            lambda f: f.name in facet_names and getvalue(query, f.name, None) is not None,
            facets,
        )
    )


def infer(
    casebase: Iterable[C],
    request: ReasoningRequest[C],
    evaluator: Evaluator,
    *,
    getvalue: Callable[[Any, str, Optional[Any]], Any] = getattr,
) -> ReasoningResponse[C]:

    threshold = request.threshold
    evaluate_cases = map(
        lambda c: Result(evaluator(request.query, c), c),
        casebase,
    )
    sort_results = sorted(
        evaluate_cases,
        key=lambda r: r.similarity,
        reverse=True,
    )
    calculate_results = filter(
        lambda r: r.similarity >= threshold,
        sort_results,
    )
    if request.facets is not None:
        relevant_facets = _make_relevant_facets(request.query, request.facets, getvalue)
        calculate_results = FacetCollectingIterator(calculate_results, relevant_facets, getvalue=getvalue)
    result_list = list(calculate_results)
    total_number_of_hits = len(result_list)

    offset = request.offset
    limit = request.limit
    result_list = result_list[offset:offset + limit]  # fmt: skip
    return ReasoningResponse(
        total_number_of_hits,
        hits=result_list,
        facets=calculate_results.facets if request.facets is not None else None,
    )
