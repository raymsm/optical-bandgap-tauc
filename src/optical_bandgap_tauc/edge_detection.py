"""Linear absorption-edge regime detection algorithms.

This module implements the core algorithms for locating the linear region
of the absorption edge in Tauc plots, including derivative-plateau search
and curvature-based cross-checks.
"""

from dataclasses import dataclass
import logging
import numpy as np

from optical_bandgap_tauc.fitting import fit_linear_regression
from optical_bandgap_tauc.tauc import smooth_tauc_data

logger = logging.getLogger(__name__)


@dataclass
class LinearRegimeResult:
    """Dataclass holding results of the linear-regime detection and fit."""

    window_start: float
    window_end: float
    method: str
    slope: float
    intercept: float
    r_squared: float
    band_gap_ev: float
    # Cross-check details
    cross_window_start: float | None = None
    cross_window_end: float | None = None
    cross_r_squared: float | None = None
    cross_band_gap_ev: float | None = None
    disagreement_ev: float | None = None


def detect_linear_regime(
    E: np.ndarray,
    y: np.ndarray,
    derivative_threshold_pct: float = 90.0,
    curvature_threshold_pct: float = 10.0,
    min_window_points: int = 5,
    fallback_window_ev: float = 0.15,
    disagreement_warn_ev: float = 0.05,
    smooth_window: int | None = None,
    smooth_order: int | None = None,
    edge_window_ev: float | None = None,
) -> LinearRegimeResult:
    """Detect the linear absorption-edge regime in Tauc data and perform regression.

    Parameters
    ----------
    E : np.ndarray
        Array of photon energy values in eV (sorted ascending).
    y : np.ndarray
        Array of Tauc variable values y_n(E) corresponding to energy E.
    derivative_threshold_pct : float, optional
        Percentage of maximum first derivative for plateau boundary search.
        Default is 90.0.
    curvature_threshold_pct : float, optional
        Percentage of maximum absolute second derivative defining a clear inflection.
        Default is 10.0.
    min_window_points : int, optional
        Minimum number of points required in the candidate linear window.
        Default is 5.
    fallback_window_ev : float, optional
        Span in eV to expand around E_peak if window contains too few points.
        Default is 0.15.
    disagreement_warn_ev : float, optional
        Energy threshold in eV to trigger warnings about method disagreements.
        Default is 0.05.
    smooth_window : int | None, optional
        User-specified smoothing window size for Savitzky-Golay filter.
    smooth_order : int | None, optional
        User-specified polynomial order for Savitzky-Golay filter.
    edge_window_ev : float | None, optional
        Manual override of detected window width (skips auto-detection and centers on E_peak).

    Returns
    -------
    LinearRegimeResult
        Dataclass containing fit parameters, bounds, and cross-check results.
    """
    N = len(E)
    if N < 2:
        raise ValueError("At least 2 data points are required to detect a linear regime.")

    # 1. Smooth the initial Tauc variable curve
    y_smoothed = smooth_tauc_data(y, user_window=smooth_window, user_order=smooth_order)

    # 2. Compute first derivative (non-uniform E spacing handled by np.gradient)
    dy_dE = np.gradient(y_smoothed, E)

    # 3. Smooth the first derivative to ensure second derivative stability
    dy_dE_smoothed = smooth_tauc_data(
        dy_dE, user_window=smooth_window, user_order=smooth_order
    )

    # 4. Compute second derivative
    d2y_dE2 = np.gradient(dy_dE_smoothed, E)

    # 5. Candidate edge location (E_peak where dy/dE is max)
    i_peak = int(np.argmax(dy_dE_smoothed))
    E_peak = E[i_peak]

    # Check for manual window width override
    if edge_window_ev is not None:
        method = "manual-override"
        logger.debug(f"Manual override: centering {edge_window_ev} eV window at peak {E_peak:.3f} eV.")
        w_start_ev = E_peak - edge_window_ev / 2.0
        w_end_ev = E_peak + edge_window_ev / 2.0
        indices = np.where((E >= w_start_ev) & (E <= w_end_ev))[0]
        if len(indices) < 2:
            i_start = max(0, i_peak - min_window_points // 2)
            i_end = min(N - 1, i_start + min_window_points - 1)
        else:
            i_start = int(indices[0])
            i_end = int(indices[-1])
    else:
        # 6. Plateau search (peak derivative check)
        threshold = (derivative_threshold_pct / 100.0) * dy_dE_smoothed[i_peak]
        i_start = i_peak
        i_end = i_peak

        while i_start > 0 and dy_dE_smoothed[i_start - 1] >= threshold:
            i_start -= 1
        while i_end < N - 1 and dy_dE_smoothed[i_end + 1] >= threshold:
            i_end += 1

        # 7. Second-derivative sanity check (inflection trimming)
        max_abs_d2y = np.max(np.abs(d2y_dE2)) if len(d2y_dE2) > 0 else 1.0
        curv_threshold_val = (curvature_threshold_pct / 100.0) * max_abs_d2y

        # Trim to the right (higher energy)
        for j in range(i_peak, i_end):
            if d2y_dE2[j] * d2y_dE2[j + 1] <= 0:
                # Significant zero-crossing check
                if (
                    np.abs(d2y_dE2[j]) > curv_threshold_val
                    or np.abs(d2y_dE2[j + 1]) > curv_threshold_val
                ):
                    i_end = j
                    logger.debug(f"Trimmed window right boundary to index {j} (E={E[j]:.3f} eV) due to inflection.")
                    break

        # Trim to the left (lower energy)
        for j in range(i_peak - 1, i_start - 1, -1):
            if d2y_dE2[j] * d2y_dE2[j + 1] <= 0:
                # Significant zero-crossing check
                if (
                    np.abs(d2y_dE2[j]) > curv_threshold_val
                    or np.abs(d2y_dE2[j + 1]) > curv_threshold_val
                ):
                    i_start = j + 1
                    logger.debug(f"Trimmed window left boundary to index {j+1} (E={E[j+1]:.3f} eV) due to inflection.")
                    break

        # 8. Minimum window enforcement
        method = "derivative-plateau"
        if (i_end - i_start + 1) < min_window_points:
            method = "fallback"
            logger.debug(
                f"Window size ({i_end - i_start + 1} pts) is less than min_window_points ({min_window_points}). "
                "Applying fallback window expansion."
            )
            w_start_ev = E_peak - fallback_window_ev / 2.0
            w_end_ev = E_peak + fallback_window_ev / 2.0
            indices = np.where((E >= w_start_ev) & (E <= w_end_ev))[0]
            if len(indices) < 2:
                i_start = max(0, i_peak - min_window_points // 2)
                i_end = min(N - 1, i_start + min_window_points - 1)
            else:
                i_start = int(indices[0])
                i_end = int(indices[-1])

    # 9. Main Fit
    x_fit = E[i_start : i_end + 1]
    y_fit = y_smoothed[i_start : i_end + 1]
    slope, intercept, r_squared, band_gap_ev = fit_linear_regression(x_fit, y_fit)

    # 10. Anchored Curvature Cross-Check Method
    # Restrict search for max second derivative to region where dy_dE >= 15% of max
    cc_mask = dy_dE_smoothed >= 0.15 * dy_dE_smoothed[i_peak]
    cc_indices = np.where(cc_mask)[0]

    if len(cc_indices) == 0:
        cc_indices = np.array([i_peak])

    # Find the index of maximum upward curvature in the restricted region
    i_curve_max = cc_indices[int(np.argmax(d2y_dE2[cc_indices]))]
    E_curve_max = E[i_curve_max]

    # Perform linear regression in a window of width fallback_window_ev starting at E_curve_max
    cc_w_mask = (E >= E_curve_max) & (E <= E_curve_max + fallback_window_ev)
    cc_w_indices = np.where(cc_w_mask)[0]

    if len(cc_w_indices) < 2:
        cc_w_indices = np.arange(i_curve_max, min(N, i_curve_max + min_window_points))

    x_cc = E[cc_w_indices]
    y_cc = y_smoothed[cc_w_indices]
    slope_cc, intercept_cc, r_squared_cc, band_gap_cc = fit_linear_regression(x_cc, y_cc)

    # Calculate disagreement warning
    disagreement = abs(band_gap_ev - band_gap_cc)
    if disagreement > disagreement_warn_ev:
        logger.warning(
            f"Disagreement warning: Main method Eg ({band_gap_ev:.3f} eV) and "
            f"Cross-check method Eg ({band_gap_cc:.3f} eV) differ by {disagreement:.3f} eV "
            f"(limit: {disagreement_warn_ev} eV)."
        )

    return LinearRegimeResult(
        window_start=float(E[i_start]),
        window_end=float(E[i_end]),
        method=method,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        band_gap_ev=band_gap_ev,
        cross_window_start=float(E[cc_w_indices[0]]),
        cross_window_end=float(E[cc_w_indices[-1]]),
        cross_r_squared=r_squared_cc,
        cross_band_gap_ev=band_gap_cc,
        disagreement_ev=disagreement,
    )
