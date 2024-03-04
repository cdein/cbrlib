from typing import Any, Callable, Mapping

Evaluator = Callable[[Any, Any], float]


def equality(query: Any, case: Any) -> float:
    if query != case:
        return 0
    return 1


def total_order(
    ordering: list[Any], evaluate: Evaluator, query: Any, case: Any
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
