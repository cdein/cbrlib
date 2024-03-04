from cbrlib import evaluate

LOOKUP = {
    "red": {"red": 1, "orange": 0.8, "yellow": 0.4},
    "orange": {"orange": 1, "red": 0.8, "yellow": 0.8},
}


def test_table_lookup_similarity() -> None:
    assert evaluate.table_lookup(LOOKUP, "red", "red") == 1
    assert evaluate.table_lookup(LOOKUP, "red", "orange") == 0.8
    assert evaluate.table_lookup(LOOKUP, "red", "yellow") == 0.4

    assert evaluate.table_lookup(LOOKUP, "orange", "red") == 0.8
    assert evaluate.table_lookup(LOOKUP, "orange", "orange") == 1
    assert evaluate.table_lookup(LOOKUP, "orange", "yellow") == 0.8


def test_table_lookup_missing_value() -> None:
    assert evaluate.table_lookup(LOOKUP, "red", "green") == 0
    assert evaluate.table_lookup(LOOKUP, "green", "red") == 0
