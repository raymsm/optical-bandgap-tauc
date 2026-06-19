"""Unit tests for linear fitting and bandgap extrapolation calculations."""

import numpy as np
import pytest

from optical_bandgap_tauc.fitting import fit_linear_regression, get_window_data


def test_fit_linear_regression_perfect():
    """Test regression on a perfect straight line."""
    # Line: y = 2x - 4 -> Eg = -(-4)/2 = 2.0 eV
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x - 4.0

    slope, intercept, r_squared, Eg = fit_linear_regression(x, y)

    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, -4.0)
    assert np.isclose(r_squared, 1.0)
    assert np.isclose(Eg, 2.0)


def test_fit_linear_regression_degenerate():
    """Test regression with fewer than 2 points or zero slope."""
    # Fewer than 2 points
    x_short = np.array([1.0])
    y_short = np.array([2.0])
    assert fit_linear_regression(x_short, y_short) == (0.0, 0.0, 0.0, 0.0)

    # Empty array
    assert fit_linear_regression(np.array([]), np.array([])) == (0.0, 0.0, 0.0, 0.0)

    # Zero slope: y = 3 (Eg is undefined/infinity, should handle division by zero)
    x_zero = np.array([1.0, 2.0, 3.0])
    y_zero = np.array([3.0, 3.0, 3.0])
    slope, intercept, r_squared, Eg = fit_linear_regression(x_zero, y_zero)
    assert slope == 0.0
    assert Eg == 0.0


def test_get_window_data():
    """Test slicing of energy and data arrays within energy boundaries."""
    energy = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    y = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

    # Slice between 1.4 and 2.6 eV
    ew, yw = get_window_data(energy, y, 1.4, 2.6)
    assert np.array_equal(ew, np.array([1.5, 2.0, 2.5]))
    assert np.array_equal(yw, np.array([15.0, 20.0, 25.0]))
