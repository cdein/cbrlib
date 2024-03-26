import math
from typing import Any, Callable, Iterable, Iterator, Optional

from cbrlib.types import Facet, FacetConfig, FacetValueOrderCriteria, FacetValue, Result


class FacetCollectingIterator(Iterator):
    def __init__(
        self,
        iterable: Iterable[Result],
        facets: Iterable[FacetConfig],
        *,
        getvalue: Callable[[Any, str, Optional[Any]], Any] = getattr
    ) -> None:
        self._iterator = iter(iterable)
        self._facets = facets
        self._getvalue = getvalue
        self._facet_names = [f.name for f in facets]
        self._facet_collection = {}
        self._divider = 0
        self._value_statistics = {}

    def __next__(self) -> Result:
        result: Result = next(self._iterator)
        self._divider += 1
        weight = result.similarity
        case = result.case
        for facet_name in self._facet_names:
            value = self._getvalue(case, facet_name)
            if value is None:
                continue
            value_cache: dict = self._facet_collection.setdefault(facet_name, {})
            facet_value: FacetValue = value_cache.setdefault(value, FacetValue(value))
            facet_value.count += 1
            facet_value.importance += weight
        return result

    @property
    def facets(self) -> Iterable[Facet]:
        return sorted(
            [
                Facet(
                    facet.name,
                    _to_facet_values(
                        self._facet_collection[facet.name].values(),
                        facet.order_by,
                        facet.max_count,
                    ),
                    _calculate_entropy(self._facet_collection[facet.name].values(), self._divider),
                )
                for facet in self._facets
                if facet.name in self._facet_collection
            ],
            key=lambda f: f.entropy,
            reverse=True,
        )


def _to_facet_values(
    values: Iterable[FacetValue],
    order_criteria: FacetValueOrderCriteria,
    max_count: int,
) -> list[FacetValue]:
    sorted_values = sorted(
        _apply_facet_value_importance(values),
        key=order_criteria.as_key_function(),
        reverse=order_criteria.is_reverse(),
    )
    return sorted_values[0:max_count]  # fmt: skip


def _calculate_entropy(facet_values: Iterable[FacetValue], divider: int) -> float:
    return sum([-((value.count / divider) * math.log2(value.count / divider)) for value in facet_values])


def _apply_facet_value_importance(
    facet_values: Iterable[FacetValue],
) -> Iterator[FacetValue]:
    for facet_value in facet_values:
        facet_value.importance /= facet_value.count
        yield facet_value
