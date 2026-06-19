"""Tauc transform computation and spectral smoothing.

This module provides functions to calculate Tauc variables for direct/indirect
allowed/forbidden transitions and apply Savitzky-Golay filters.
"""

import logging
import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

TRANSITIONS = {
    "direct-allowed": 2.0,
    "indirect-allowed": 0.5,
    "direct-forbidden": 2.0 / 3.0,
    "indirect-forbidden": 1.0 / 3.0,
}


def nearest_odd(value: float) -> int:
    """Round a value to the nearest odd integer, with a floor of 1.

    Parameters
    ----------
    value : float
        The input floating point value.

    Returns
    -------
    int
        The nearest odd integer >= 1.
    """
    rounded = int(round(value))
    rounded = max(1, rounded)
    if rounded % 2 == 0:
        rounded += 1
    return rounded


def compute_savgol_params(
    n_points: int,
    user_window: int | None = None,
    user_order: int | None = None,
) -> tuple[int, int, bool]:
    """Compute adaptive Savitzky-Golay window and polynomial order parameters.

    Parameters
    ----------
    n_points : int
        Number of data points in the spectrum.
    user_window : int | None, optional
        User-specified smoothing window length.
    user_order : int | None, optional
        User-specified polynomial order.

    Returns
    -------
    tuple[int, int, bool]
        A tuple of (window_length, polyorder, should_smooth). If should_smooth is
        False, the data should not be smoothed.
    """
    if n_points < 3:
        return 1, 0, False

    # Default polynomial order
    order = user_order if user_order is not None else 3

    # Default window length
    if user_window is not None:
        # Ensure user window is odd
        window = user_window
        if window % 2 == 0:
            window = max(1, window - 1)
    else:
        window = nearest_odd(0.05 * n_points)
        window = max(5, window)  # Default minimum window to support order=3

    # Clamp the upper end to prevent out-of-bounds on short datasets
    max_allowed = n_points if n_points % 2 == 1 else n_points - 1
    window = min(window, max_allowed)

    # Adjust polynomial order if window is too small
    if order >= window:
        if window > 1:
            order = window - 1
        else:
            return 1, 0, False

    # Final check to verify window is odd and greater than order
    if window % 2 == 0 or window <= order:
        return window, order, False

    return window, order, True


def smooth_tauc_data(
    y: np.ndarray,
    user_window: int | None = None,
    user_order: int | None = None,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter to the Tauc data.

    Parameters
    ----------
    y : np.ndarray
        The input Tauc variable y_n(E).
    user_window : int | None, optional
        User-specified smoothing window length.
    user_order : int | None, optional
        User-specified polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed Tauc data array.
    """
    n_points = len(y)
    window, order, should_smooth = compute_savgol_params(n_points, user_window, user_order)

    if not should_smooth:
        logger.debug("Smoothing skipped due to insufficient data points or small window.")
        return np.copy(y)

    logger.debug(f"Applying Savitzky-Golay filter: window_length={window}, polyorder={order}")
    return savgol_filter(y, window, order)


def compute_tauc_variable(
    energy: np.ndarray,
    f_r: np.ndarray,
    transition_type: str,
) -> np.ndarray:
    """Compute the Tauc variable y_n(E) = (F(R) * E)^n.

    Parameters
    ----------
    energy : np.ndarray
        Array of photon energy values in eV.
    f_r : np.ndarray
        Array of Kubelka-Munk proxy values.
    transition_type : str
        The transition type (e.g. 'direct-allowed', 'indirect-allowed').

    Returns
    -------
    np.ndarray
        Tauc variable values.

    Raises
    ------
    ValueError
        If transition_type is not one of the allowed types.
    """
    if transition_type not in TRANSITIONS:
        raise ValueError(
            f"Unknown transition type '{transition_type}'. "
            f"Allowed types: {list(TRANSITIONS.keys())}"
        )

    n = TRANSITIONS[transition_type]
    base = f_r * energy

    # Clip negative values to zero to avoid complex numbers/NaN when taking fractional power
    neg_mask = base < 0.0
    if np.any(neg_mask):
        logger.debug(
            f"Clipping {np.sum(neg_mask)} negative (F(R)*E) points to 0.0 "
            f"for transition '{transition_type}'"
        )
        base = np.maximum(0.0, base)

    return np.power(base, n)
