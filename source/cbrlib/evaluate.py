from collections import namedtuple
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


def set_query_inclusion(evaluator: Evaluate, query: set[Any], case: set[Any]) -> float:
    size_of_query = len(query)
    if size_of_query == 0:
        return 0
    current = functools.reduce(
        lambda e1, e2: e1 + coverage(e2, case, evaluator), [0, *query]
    )
    return current / size_of_query


def set_case_inclusion(evaluator: Evaluate, query: set[Any], case: set[Any]) -> float:
    return set_query_inclusion(evaluator, case, query)


def set_intermediate(evaluator: Evaluate, query: set[Any], case: set[Any]) -> float:
    sim_1 = set_query_inclusion(evaluator, query, case)
    sim_2 = set_query_inclusion(evaluator, case, query)
    return (sim_1 + sim_2) / 2


WeightedPropertyEvaluatorMapping = namedtuple(
    "WeightedPropertyEvaluatorMapping",
    {"property_name", "evaluator", "weight"},
)


def case_average(
    mappings: Iterable[WeightedPropertyEvaluatorMapping],
    query: Any,
    case: Any,
    *,
    getvalue: Callable[[Any, str], Any] = getattr
) -> float:
    divider = 0
    similarity_sum = 0
    for mapping in mappings:
        property_name = mapping[0]
        evaluator = mapping[1]
        weight = mapping[2]
        query_value = getvalue(query, property_name)
        if query_value is None:
            continue
        divider += weight
        case_value = getvalue(case, property_name)
        similarity_sum += weight * evaluator(query_value, case_value)
    if divider <= 0:
        return 0
    return similarity_sum / divider
