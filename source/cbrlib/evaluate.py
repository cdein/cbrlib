import functools
import math
from typing import Any, Callable, Iterable, Mapping, Sequence
from statistics import median

from cbrlib.types import Evaluator, NumericEvaluationOptions, PropertyEvaluatorMapping, WeightedPropertyEvaluatorMapping


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


def _collect_similarities(
    mappings: Iterable[PropertyEvaluatorMapping],
    query: Any,
    case: Any,
    getvalue: Callable[[Any, str], Any],
) -> list[float]:
    similarity_results = []
    for mapping in mappings:
        property_name = mapping[0]
        evaluator = mapping[1]
        query_value = getvalue(query, property_name)
        if query_value is None:
            continue
        case_value = getvalue(case, property_name)
        similarity = evaluator(query_value, case_value)
        similarity_results.append(similarity)
    return similarity_results


def case_median(
    mappings: Iterable[PropertyEvaluatorMapping],
    query: Any,
    case: Any,
    *,
    getvalue: Callable[[Any, str], Any] = getattr
) -> float:
    similarity_results = _collect_similarities(mappings, query, case, getvalue)
    if not similarity_results:
        return 0
    return median(sorted(similarity_results))


def case_min(
    mappings: Iterable[PropertyEvaluatorMapping],
    query: Any,
    case: Any,
    *,
    getvalue: Callable[[Any, str], Any] = getattr
) -> float:
    similarity_results = _collect_similarities(mappings, query, case, getvalue)
    if not similarity_results:
        return 0
    return min(similarity_results)


def case_max(
    mappings: Iterable[PropertyEvaluatorMapping],
    query: Any,
    case: Any,
    *,
    getvalue: Callable[[Any, str], Any] = getattr
) -> float:
    similarity_results = _collect_similarities(mappings, query, case, getvalue)
    if not similarity_results:
        return 0
    return max(similarity_results)


def case_euclidean(
    mappings: Iterable[PropertyEvaluatorMapping],
    query: Any,
    case: Any,
    getvalue: Callable[[object, str], Any] = getattr,
) -> float:
    similarity_sum = 0
    for mapping in mappings:
        property_name = mapping[0]
        evaluator = mapping[1]
        query_value = getvalue(query, property_name)
        if query_value is None:
            continue
        case_value = getvalue(case, property_name)
        similarity = evaluator(query_value, case_value)
        if similarity <= 0:
            continue
        similarity_sum += similarity**2
    return math.sqrt(similarity_sum)


def equality(query: Any, case: Any) -> float:
    if query != case:
        return 0
    return 1


def total_order(ordering: Sequence[Any], evaluate: Evaluator, query: Any, case: Any) -> float:
    try:
        query_index = ordering.index(query)
        case_index = ordering.index(case)
    except ValueError:
        return 0
    else:
        return evaluate(query_index, case_index)


def table_lookup(lookup: Mapping[str, Mapping[str, float]], query: Any, case: Any) -> float:
    if query not in lookup:
        return 0
    query_map = lookup[query]
    if case not in query_map:
        return 0
    return query_map[case]


def coverage(query: Any, bulk: Iterable[Any], evaluate: Evaluator = equality) -> float:
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


def set_query_inclusion(evaluator: Evaluator, query: Sequence[Any], case: Sequence[Any]) -> float:
    size_of_query = len(query)
    if size_of_query == 0:
        return 0
    current = functools.reduce(lambda e1, e2: e1 + coverage(e2, case, evaluator), [0, *query])
    return current / size_of_query


def set_case_inclusion(evaluator: Evaluator, query: Sequence[Any], case: Sequence[Any]) -> float:
    return set_query_inclusion(evaluator, case, query)


def set_intermediate(evaluator: Evaluator, query: Sequence[Any], case: Sequence[Any]) -> float:
    sim_1 = set_query_inclusion(evaluator, query, case)
    sim_2 = set_query_inclusion(evaluator, case, query)
    return (sim_1 + sim_2) / 2


def _calculate_distance(v1: float, v2: float, max_distance: float, cyclic: bool) -> float:
    result = abs(v1 - v2)
    if cyclic and result > max_distance:
        result = 2 * max_distance - result
    return result


def _calculate_max_distance(v: float, max_distance: float, origin: float, use_origin: bool) -> float:
    if use_origin:
        return abs(v - origin)
    return max_distance


def _is_less(v1: float, v2: float, max_distance: float, cyclic: bool) -> bool:
    if not cyclic:
        return v1 < v2

    if v1 < v2:
        left_distance = v2 - v1
    else:
        left_distance = 2 * max_distance - v1 + v2

    right_distance = 2 * max_distance - left_distance
    return left_distance < right_distance


def numeric(options: NumericEvaluationOptions, query: float, case: float) -> float:
    max_distance = _calculate_max_distance(query, options.max_distance, options.origin, options.use_origin)
    if max_distance == 0:
        return 1.0

    distance = _calculate_distance(query, case, options.max_distance, options.cyclic)
    relative_distance = distance / max_distance
    if relative_distance >= 1:
        return 0.0

    less = _is_less(case, query, options.max_distance, options.cyclic)
    parameters = options.if_less if less else options.if_more

    if relative_distance <= parameters.equal:
        return 1.0
    elif relative_distance >= parameters.tolerance:
        return 0.0

    stretched_distance = (relative_distance - parameters.equal) / (parameters.tolerance - parameters.equal)
    interpolation = parameters.get_interpolation()
    return interpolation(stretched_distance, parameters.linearity)
