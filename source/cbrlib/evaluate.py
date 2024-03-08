import functools
from typing import Any, Callable, Iterable, Mapping

Evaluate = Callable[[Any, Any], float]


def equality(query: Any, case: Any) -> float:
    if query != case:
        return 0
    return 1


def total_order(
    ordering: list[Any], evaluate: Evaluate, query: Any, case: Any
) -> float:
    try:
        query_index = ordering.index(query)
        case_index = ordering.index(case)
    except ValueError:
        return 0
    else:
        return evaluate(query_index, case_index)


def table_lookup(
    lookup: Mapping[str, Mapping[str, float]], query: Any, case: Any
) -> float:
    if query not in lookup:
        return 0
    query_map = lookup[query]
    if case not in query_map:
        return 0
    return query_map[case]


def coverage(query: Any, bulk: Iterable[Any], evaluate: Evaluate = equality) -> float:
    similarity_sum = 0
    element_count = 0
    for element in bulk:
        similarity = evaluate(query, element)
        if similarity == 1:
            return 1
        similarity_sum += similarity
        element_count += 1
    if element_count == 0:
        return 0
    return similarity_sum / element_count


def set_query_inclusion(query: set[Any], case: set[Any], evaluator: Evaluate) -> float:
    size_of_query = len(query)
    if size_of_query == 0:
        return 0
    current = functools.reduce(
        lambda e1, e2: e1 + coverage(e2, case, evaluator), [0, *query]
    )
    return float(current) / size_of_query


def set_case_inclusion(query: set[Any], case: set[Any], evaluator: Evaluate) -> float:
    return set_query_inclusion(case, query, evaluator)


def set_intermediate(query: set[Any], case: set[Any], evaluator: Evaluate) -> float:
    sim_1 = set_query_inclusion(query, case, evaluator)
    sim_2 = set_query_inclusion(case, query, evaluator)
    return (sim_1 + sim_2) / 2
