import pytest

from cbrlib import (
    FunctionCalculationParameter,
    NumericEvaluationOptions,
    NumericInterpolation,
    types,
)
from cbrlib import evaluate


def test_calculate_distance() -> None:
    assert evaluate._calculate_distance(7, 15, 7, False) == 8
    assert evaluate._calculate_distance(7, 15, 7, True) == 6


def test_calculate_max_distance() -> None:
    assert evaluate._calculate_max_distance(7, 15, 0, False) == 15
    assert evaluate._calculate_max_distance(7, 15, 2, False) == 15
    assert evaluate._calculate_max_distance(7, 15, 0, True) == 7
    assert evaluate._calculate_max_distance(7, 15, 2, True) == 5


def test_is_less() -> None:
    assert evaluate._is_less(1, 2, 3, False)
    assert not evaluate._is_less(2, 1, 3, False)

    assert evaluate._is_less(3, 4, 3, True)
    assert not evaluate._is_less(4, 3, 3, True)


def test_interpolate_polynom_linearity_zero() -> None:
    assert types._interpolate_polynom(0.8, 0) == 0


def test_interpolate_polynom_linearity_one() -> None:
    for i in [i / 10 for i in range(11)]:
        assert types._interpolate_polynom(i, 1) == 1 - i


def test_interpolate_polynom() -> None:
    expected_values = [
        1.0,
        0.81,
        0.64,
        0.49,
        0.36,
        0.25,
        0.16,
        0.09,
        0.04,
        0.01,
        0.0,
    ]
    for i in range(11):
        assert types._interpolate_polynom(i / 10, 0.5) == pytest.approx(expected_values[i])


def test_interpolate_root_linearity_zero() -> None:
    assert types._interpolate_root(0.8, 0) == 1


def test_interpolate_root_linearity_one() -> None:
    for i in [i / 10 for i in range(11)]:
        assert types._interpolate_root(i, 1) == 1 - i


def test_interpolate_root() -> None:
    expected_values = [
        1.0,
        0.95,
        0.89,
        0.84,
        0.77,
        0.71,
        0.63,
        0.55,
        0.45,
        0.315,
        0.0,
    ]
    for i in range(11):
        assert types._interpolate_root(i / 10, 0.5) == pytest.approx(
            expected_values[i], rel=1e-2
        ), f"Value at index {i} doesn't fit to expectation."


def test_interpolate_sigmoid_linearity_zero() -> None:
    assert types._interpolate_sigmoid(0.8, 0) == 0
    assert types._interpolate_sigmoid(0.4, 0) == 1


def test_interpolate_sigmoid_linearity_one() -> None:
    for i in [i / 10 for i in range(11)]:
        assert types._interpolate_sigmoid(i, 1) == 1 - i


def test_interpolate_sigmoid() -> None:
    expected_values = [
        1.0,
        0.98,
        0.92,
        0.82,
        0.68,
        0.5,
        0.32,
        0.18,
        0.08,
        0.02,
        0.0,
    ]
    for i in range(11):
        assert types._interpolate_sigmoid(i / 10, 0.5) == pytest.approx(
            expected_values[i]
        ), f"Value at index {i} doesn't fit to expectation."


def test_function_calculation_paramter() -> None:
    parameters = FunctionCalculationParameter.default()
    assert parameters.equal == 0
    assert parameters.tolerance == 0.5
    assert parameters.linearity == 1
    assert parameters.interpolation == NumericInterpolation.POLYNOM


def test_function_calculation_paramter_get_interpolation() -> None:
    parameters = FunctionCalculationParameter(
        interpolation=NumericInterpolation.POLYNOM,
    )
    assert parameters.interpolation == NumericInterpolation.POLYNOM
    assert parameters.get_interpolation() is types._interpolate_polynom
    parameters = FunctionCalculationParameter(
        interpolation=NumericInterpolation.ROOT,
    )
    assert parameters.interpolation == NumericInterpolation.ROOT
    assert parameters.get_interpolation() is types._interpolate_root
    parameters = FunctionCalculationParameter(interpolation=NumericInterpolation.SIGMOID)
    assert parameters.interpolation == NumericInterpolation.SIGMOID
    assert parameters.get_interpolation() is types._interpolate_sigmoid


def test_evaluation_options() -> None:
    options = NumericEvaluationOptions(20, 0)
    assert options.min_ == 0
    assert options.max_ == 20
    assert options.origin == 0
    assert not options.cyclic
    assert not options.use_origin
    assert options.max_distance == 20
    assert options.if_less is FunctionCalculationParameter.default()
    assert options.if_more is FunctionCalculationParameter.default()


def test_numeric_evaluation_no_distance() -> None:
    options = NumericEvaluationOptions(10, 10)
    assert evaluate.numeric(options, 10, 20) == 1
    assert evaluate.numeric(options, 0, 20) == 1
    assert evaluate.numeric(options, 10, 10) == 1
    assert evaluate.numeric(options, 100, 0) == 1


def test_numeric_evaluation_unexpected_distance() -> None:
    options = NumericEvaluationOptions(0, 10)
    assert evaluate.numeric(options, 10, 20) == 0
    assert evaluate.numeric(options, 0, 20) == 0
    assert evaluate.numeric(options, 100, 0) == 0


def test_numeric_evaluation_equality() -> None:
    options = NumericEvaluationOptions(0, 10, if_less=FunctionCalculationParameter(equal=0.1))
    for i in range(11):
        assert evaluate.numeric(options, 10, 9 + (i / 10)) == 1
    assert evaluate.numeric(options, 8.9999, 10) < 1
    options = NumericEvaluationOptions(0, 10, if_more=FunctionCalculationParameter(equal=0.1))
    for i in range(11):
        assert evaluate.numeric(options, 9 + (i / 10), 10) == 1
    assert evaluate.numeric(options, 10, 8.9999) < 1


def test_numeric_evaluation_tolerance() -> None:
    options = NumericEvaluationOptions(0, 10, if_less=FunctionCalculationParameter(tolerance=0.1))
    assert evaluate.numeric(options, 10, 9.0001) > 0
    for i in range(9, -1, -1):
        assert evaluate.numeric(options, 10, i) == 0
    options = NumericEvaluationOptions(0, 10, if_more=FunctionCalculationParameter(tolerance=0.1))
    assert evaluate.numeric(options, 9.0001, 10) > 0
    for i in range(9, -1, -1):
        assert evaluate.numeric(options, i, 10) == 0


def test_numeric_evaluation() -> None:
    expected_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    options = NumericEvaluationOptions(0, 20)
    for i in range(1, 11):
        assert evaluate.numeric(options, i, 10) == pytest.approx(
            expected_values[i - 1],
        )
