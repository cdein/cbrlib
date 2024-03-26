from cbrlib import evaluate

lookup = {
    "red": {"orange": 0.8, "yellow": 0.4},
    "orange": {"red": 0.8, "yellow": 0.8},
}


def test_table_lookup_similarity() -> None:
    assert evaluate.table_lookup(lookup, "red", "red") == 1
    assert evaluate.table_lookup(lookup, "red", "orange") == 0.8
    assert evaluate.table_lookup(lookup, "red", "yellow") == 0.4

    assert evaluate.table_lookup(lookup, "orange", "red") == 0.8
    assert evaluate.table_lookup(lookup, "orange", "orange") == 1
    assert evaluate.table_lookup(lookup, "orange", "yellow") == 0.8

    assert evaluate.table_lookup(lookup, "yellow", "yellow") == 1

    assert evaluate.table_lookup(lookup, "green", "green") == 1


def test_table_lookup_missing_value() -> None:
    assert evaluate.table_lookup(lookup, "red", "green") == 0
    assert evaluate.table_lookup(lookup, "green", "red") == 0
