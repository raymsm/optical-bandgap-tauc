"""Kubelka-Munk conversion and wavelength-to-energy calculations.

This module provides functions to convert reflectance to Kubelka-Munk proxy values
and to convert wavelength measurements in nm to photon energy in eV.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

HC_CONVERSION_FACTOR = 1239.84198  # Planck's const * speed of light in eV*nm


def wavelength_to_energy(wavelength: np.ndarray) -> np.ndarray:
    """Convert wavelength in nm to photon energy in eV.

    Parameters
    ----------
    wavelength : np.ndarray
        Array of wavelength values in nanometers (nm).

    Returns
    -------
    np.ndarray
        Array of photon energy values in electron-volts (eV).

    Raises
    ------
    ValueError
        If any wavelength value is less than or equal to zero.
    """
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength values must be strictly positive (greater than 0 nm).")
    return HC_CONVERSION_FACTOR / wavelength


def sort_by_energy(energy: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort energy and corresponding data arrays in ascending order of energy.

    Parameters
    ----------
    energy : np.ndarray
        Array of photon energy values.
    data : np.ndarray
        Array of corresponding spectral data (e.g. reflectance, absorbance).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (sorted_energy, sorted_data).
    """
    sort_indices = np.argsort(energy)
    return energy[sort_indices], data[sort_indices]


def convert_and_sort(wavelength: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert wavelength to energy and sort both by energy ascending.

    Parameters
    ----------
    wavelength : np.ndarray
        Array of wavelength values in nm.
    data : np.ndarray
        Array of corresponding spectral data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (energy_sorted, data_sorted) in eV.
    """
    energy = wavelength_to_energy(wavelength)
    return sort_by_energy(energy, data)


def reflectance_to_kubelka_munk(
    reflectance: np.ndarray, clip_threshold: float = 1e-10
) -> np.ndarray:
    """Convert reflectance (fraction 0-1) to the Kubelka-Munk proxy function F(R).

    F(R) = (1 - R)^2 / (2R)

    If reflectance values fall outside [clip_threshold, 1.0], they are clipped.
    A warning is logged if more than 1% of the data points are modified by clipping.

    Parameters
    ----------
    reflectance : np.ndarray
        Reflectance values as fractions (0.0 to 1.0).
    clip_threshold : float, optional
        Minimum value to clip reflectance to, preventing division by zero.
        Default is 1e-10.

    Returns
    -------
    np.ndarray
        Kubelka-Munk values F(R).
    """
    if len(reflectance) == 0:
        return np.array([], dtype=float)

    # Detect values outside the physical range [clip_threshold, 1.0]
    clipped_mask = (reflectance < clip_threshold) | (reflectance > 1.0)
    clip_count = np.sum(clipped_mask)
    clip_fraction = clip_count / len(reflectance)

    if clip_fraction > 0.01:
        logger.warning(
            f"Clipping warning: {clip_fraction * 100:.2f}% of reflectance data points ({clip_count}/{len(reflectance)}) "
            f"were outside the range [{clip_threshold}, 1.0] and were clipped. "
            "This may indicate detector saturation, raw %R not properly scaled to fraction, or bad baseline correction."
        )

    # Clip to valid range
    r_clipped = np.clip(reflectance, clip_threshold, 1.0)
    return (1.0 - r_clipped) ** 2 / (2.0 * r_clipped)
