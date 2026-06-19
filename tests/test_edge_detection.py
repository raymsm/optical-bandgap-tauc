"""Unit tests for the linear-regime edge detection algorithm."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from optical_bandgap_tauc.edge_detection import detect_linear_regime


def generate_synthetic_tauc(
    energy: np.ndarray,
    Eg: float = 2.0,
    slope: float = 10.0,
    sat_offset: float = 0.8,
    smooth_sigma: float = 4.0,
    noise_level: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic Tauc curve (F(R)*E)^n with a known bandgap.

    Creates a sub-gap flat region, a linear rising edge, a saturation plateau,
    smoothes the transitions with a Gaussian filter, and adds Gaussian noise.

    Parameters
    ----------
    energy : np.ndarray
        Array of photon energies (eV).
    Eg : float
        Injected band gap (eV).
    slope : float
        Slope of the linear rising region.
    sat_offset : float
        Energy offset above Eg where saturation begins.
    smooth_sigma : float
        Standard deviation of Gaussian filter for transition smoothing.
    noise_level : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for noise reproducibility.

    Returns
    -------
    np.ndarray
        Synthetic Tauc plot y-values.
    """
    # 1. Piecewise baseline, rising edge, and saturation plateau
    y = np.zeros_like(energy)
    sat_energy = Eg + sat_offset

    # Rising edge
    mask_rise = (energy >= Eg) & (energy <= sat_energy)
    y[mask_rise] = slope * (energy[mask_rise] - Eg)

    # Saturation plateau
    mask_sat = energy > sat_energy
    y[mask_sat] = slope * sat_offset

    # 2. Smooth transition using Gaussian filter to make derivatives smooth and realistic
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)

    # 3. Add noise
    if noise_level > 0.0:
        np.random.seed(seed)
        noise = np.random.normal(0.0, noise_level, size=len(energy))
        y_smooth = y_smooth + noise

    return np.maximum(0.0, y_smooth)


def test_detect_linear_regime_noise_levels():
    """Test bandgap recovery from synthetic data at multiple noise levels."""
    # Typical energy range for a visible band gap (1.5 - 3.5 eV)
    energy = np.linspace(1.5, 3.5, 200)
    injected_Eg = 2.20

    # 1. Zero Noise
    y_clean = generate_synthetic_tauc(energy, Eg=injected_Eg, noise_level=0.0)
    res_clean = detect_linear_regime(
        energy, y_clean, smooth_window=15, smooth_order=3
    )
    # With zero noise, we should recover Eg within extremely tight tolerance (0.02 eV)
    assert res_clean.r_squared > 0.99
    assert abs(res_clean.band_gap_ev - injected_Eg) < 0.02

    # 2. Low Noise (1%)
    y_low = generate_synthetic_tauc(energy, Eg=injected_Eg, noise_level=0.1)
    res_low = detect_linear_regime(
        energy, y_low, smooth_window=15, smooth_order=3
    )
    assert res_low.r_squared > 0.95
    assert abs(res_low.band_gap_ev - injected_Eg) < 0.05

    # 3. Medium Noise (3%)
    y_med = generate_synthetic_tauc(energy, Eg=injected_Eg, noise_level=0.3)
    res_med = detect_linear_regime(
        energy, y_med, smooth_window=15, smooth_order=3
    )
    # Higher noise might degrade accuracy slightly, but should still be close
    assert abs(res_med.band_gap_ev - injected_Eg) < 0.10


def test_edge_case_pure_noise():
    """Verify that pure noise does not cause a crash and is flagged correctly."""
    energy = np.linspace(1.5, 3.5, 100)
    np.random.seed(123)
    y_noise = np.random.normal(0.5, 0.2, size=len(energy))
    y_noise = np.maximum(0.0, y_noise)

    # Should run and not raise an exception, though fit quality R2 will be very poor
    res = detect_linear_regime(energy, y_noise)
    assert res.r_squared < 0.5


def test_edge_case_monotonically_flat():
    """Verify flat line behaviour."""
    energy = np.linspace(1.5, 3.5, 100)
    y_flat = np.ones_like(energy) * 5.0

    # Flat line means derivatives are zero, should fallback or handle gracefully
    res = detect_linear_regime(energy, y_flat)
    assert res.slope == 0.0 or res.r_squared < 0.5


def test_edge_case_very_short_spectrum():
    """Verify fallback when spectrum is too short."""
    energy = np.array([2.0, 2.1, 2.2, 2.3])
    y = np.array([0.1, 0.5, 1.2, 1.8])

    # Should run and yield a result
    res = detect_linear_regime(energy, y, min_window_points=5)
    # With fewer points than min_window_points (5), it must trigger 'fallback'
    assert res.method == "fallback"


def test_edge_case_two_edges():
    """Verify that the algorithm picks the steepest edge when two are present."""
    energy = np.linspace(1.5, 4.0, 300)

    # First shallow edge at Eg = 2.0 (slope = 5)
    # Second steep edge at Eg = 3.0 (slope = 15)
    y1 = np.maximum(0.0, 5.0 * (energy - 2.0))
    y1[energy > 2.5] = 5.0 * 0.5

    y2 = np.maximum(0.0, 15.0 * (energy - 3.0))
    y2[energy > 3.5] = 15.0 * 0.5

    y_double = y1 + y2
    y_smooth = gaussian_filter1d(y_double, sigma=3.0)

    res = detect_linear_regime(energy, y_smooth)
    # The steepest edge starts around 3.0, so the detected Eg should be close to 3.0, not 2.0
    assert abs(res.band_gap_ev - 3.0) < 0.20
