import math
from typing import Any, Callable, Iterable, Iterator, Optional

from cbrlib.types import Facet, FacetConfig, FacetOrderCriteria, FacetValue, Result


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

    def __next__(self) -> Result:
        result: Result = next(self._iterator)
        self._divider += 1
        case = result.case
        for facet_name in self._facet_names:
            value = self._getvalue(case, facet_name)
            if value is None:
                continue
            value_cache: dict = self._facet_collection.setdefault(facet_name, {})
            facet_value: FacetValue = value_cache.setdefault(value, FacetValue(value))
            facet_value.count += 1
        return result

    @property
    def facets(self) -> Iterable[Facet]:
        return sorted(
            [
                Facet(
                    facet.name,
                    _to_facet_values(self._facet_collection[facet.name].values(), facet.order_by, facet.max_count),
                    _calculate_entropy(self._facet_collection[facet.name].values(), self._divider),
                )
                for facet in self._facets
                if facet.name in self._facet_collection
            ],
            key=lambda f: f.entropy,
            reverse=True,
        )


_order_by = {
    FacetOrderCriteria.COUNT: lambda fv: fv.count,
    FacetOrderCriteria.VALUE: lambda fv: fv.value,
}


def _to_facet_values(values: Iterable[FacetValue], order_by: FacetOrderCriteria, max_count: int) -> list[FacetValue]:
    sorted_values = sorted(
        values,
        key=_order_by[order_by],
        reverse=True,
    )
    return sorted_values[0:max_count]  # fmt: skip


def _calculate_entropy(facet_values: Iterable[FacetValue], divider: int) -> float:
    return sum([-((value.count / divider) * math.log2(value.count / divider)) for value in facet_values])
