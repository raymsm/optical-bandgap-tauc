"""Linear fitting and extrapolation functions.

This module provides functions for fitting linear lines to Tauc plot regions,
calculating R^2 coefficients, and extrapolating bandgap energy values.
"""

import numpy as np
from scipy.stats import linregress


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Perform least-squares linear regression on data and calculate x-intercept.

    Fits the equation: y = slope * x + intercept
    Extrapolates the x-intercept: Eg = -intercept / slope

    Parameters
    ----------
    x : np.ndarray
        Independent variable array (photon energy in eV).
    y : np.ndarray
        Dependent variable array (smoothed Tauc variable).

    Returns
    -------
    tuple[float, float, float, float]
        A tuple of (slope, intercept, r_squared, band_gap_ev).
        Returns (0.0, 0.0, 0.0, 0.0) if regression cannot be performed
        (e.g., fewer than 2 points, zero slope).
    """
    if len(x) < 2:
        return 0.0, 0.0, 0.0, 0.0

    res = linregress(x, y)
    slope = res.slope
    intercept = res.intercept

    if slope is None or np.isnan(slope) or np.isinf(slope):
        return 0.0, 0.0, 0.0, 0.0

    # Clamp extremely small slopes to exactly zero (avoiding numerical noise)
    if abs(slope) < 1e-7:
        slope = 0.0

    r_squared = res.rvalue ** 2 if not np.isnan(res.rvalue) else 0.0

    if slope == 0.0:
        band_gap = 0.0
    else:
        band_gap = -intercept / slope

    # Handle float conversions from numpy types
    return float(slope), float(intercept), float(r_squared), float(band_gap)


def get_window_data(
    energy: np.ndarray,
    y: np.ndarray,
    start_ev: float,
    end_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract energy and Tauc variable arrays within a specified energy window.

    Parameters
    ----------
    energy : np.ndarray
        Array of energy values.
    y : np.ndarray
        Array of corresponding Tauc variable values.
    start_ev : float
        Start energy of the window (inclusive).
    end_ev : float
        End energy of the window (inclusive).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered arrays (energy_window, y_window).
    """
    mask = (energy >= start_ev) & (energy <= end_ev)
    return energy[mask], y[mask]
