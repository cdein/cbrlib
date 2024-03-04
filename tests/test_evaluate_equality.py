from cbrlib import evaluate


def test_equality() -> None:
    assert evaluate.equality(42, 42) == 1


def test_non_equality() -> None:
    assert evaluate.equality(42, 0) == 0
