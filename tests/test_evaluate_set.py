import functools
import pytest
from cbrlib import evaluate

LOOKUP = {
    "red": {"red": 1, "orange": 0.8, "yellow": 0.4},
    "orange": {"orange": 1, "red": 0.8, "yellow": 0.8},
}
LOOKUP_EVALUATOR = functools.partial(evaluate.table_lookup, LOOKUP)


def test_coverage_equality() -> None:
    assert evaluate.coverage(42, (21, 42, 84)) == 1


def test_coverage_no_equality() -> None:
    assert evaluate.coverage(43, (21, 42, 84)) == 0


def test_coverage_similarity() -> None:
    assert evaluate.coverage(
        "red",
        ("orange", "green", "purple", "yellow"),
        LOOKUP_EVALUATOR,
    ) == pytest.approx(0.3)


def test_coverage_empty_bulk() -> None:
    assert evaluate.coverage(42, []) == 0


def test_set_query_inclusion() -> None:
    assert evaluate.set_query_inclusion(
        LOOKUP_EVALUATOR,
        ("red", "orange"),
        ("yellow", "green", "purple"),
    ) == pytest.approx(0.2)


def test_set_query_inclusion_equality() -> None:
    assert (
        evaluate.set_query_inclusion(
            LOOKUP_EVALUATOR,
            ("red",),
            ("green", "red"),
        )
        == 1
    )


def test_set_query_inclusion_empty_query() -> None:
    assert (
        evaluate.set_query_inclusion(
            LOOKUP_EVALUATOR,
            [],
            ("yellow", "green", "purple"),
        )
        == 0
    )


def test_set_query_inclusion_empty_case() -> None:
    assert (
        evaluate.set_query_inclusion(
            LOOKUP_EVALUATOR,
            ("red", "orange"),
            [],
        )
        == 0
    )


def test_set_case_inclusion() -> None:
    assert evaluate.set_case_inclusion(
        LOOKUP_EVALUATOR,
        ("yellow", "green", "purple"),
        ("red", "orange"),
    ) == pytest.approx(0.2)


def test_set_case_inclusion_equality() -> None:
    assert (
        evaluate.set_case_inclusion(
            LOOKUP_EVALUATOR,
            ("green", "red"),
            ("red",),
        )
        == 1
    )


def test_set_case_inclusion_empty_query() -> None:
    assert (
        evaluate.set_case_inclusion(
            LOOKUP_EVALUATOR,
            [],
            ("yellow", "green", "purple"),
        )
        == 0
    )


def test_set_case_inclusion_empty_case() -> None:
    assert (
        evaluate.set_case_inclusion(
            LOOKUP_EVALUATOR,
            ("red", "orange"),
            [],
        )
        == 0
    )


def test_set_case_intermediate() -> None:
    assert evaluate.set_intermediate(
        LOOKUP_EVALUATOR,
        ("yellow", "green", "red"),
        ("red", "orange"),
    ) == pytest.approx(0.55)
