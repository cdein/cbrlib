from cbrlib import evaluate


def test_total_order_equality() -> None:
    assert evaluate.total_order((21, 42, 84), evaluate.equality, 42, 42) == 1


def test_total_order_non_equality() -> None:
    assert evaluate.total_order((21, 42, 84), evaluate.equality, 42, 21) == 0
    assert evaluate.total_order((21, 42, 84), evaluate.equality, 42, 84) == 0


def test_total_order_missing_value() -> None:
    assert evaluate.total_order((21, 42, 84), evaluate.equality, 42, 43) == 0
