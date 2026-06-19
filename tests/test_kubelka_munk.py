"""Unit tests for Kubelka-Munk and wavelength-energy conversions."""

import logging
import numpy as np
import pytest

from optical_bandgap_tauc.kubelka_munk import (
    wavelength_to_energy,
    sort_by_energy,
    convert_and_sort,
    reflectance_to_kubelka_munk,
)


def test_wavelength_to_energy():
    """Test standard wavelength to eV conversion and error handling."""
    # Test typical value: 1239.84198 / 500 nm = 2.47968 eV
    wl = np.array([500.0])
    expected = 1239.84198 / 500.0
    assert np.allclose(wavelength_to_energy(wl), expected)

    # Test error on negative or zero wavelength
    with pytest.raises(ValueError, match="Wavelength values must be strictly positive"):
        wavelength_to_energy(np.array([0.0]))

    with pytest.raises(ValueError, match="Wavelength values must be strictly positive"):
        wavelength_to_energy(np.array([-10.0]))


def test_sort_by_energy():
    """Test sorting array in energy-ascending order."""
    energy = np.array([3.0, 1.5, 2.0])
    data = np.array([30.0, 15.0, 20.0])

    e_sorted, d_sorted = sort_by_energy(energy, data)
    assert np.array_equal(e_sorted, np.array([1.5, 2.0, 3.0]))
    assert np.array_equal(d_sorted, np.array([15.0, 20.0, 30.0]))


def test_convert_and_sort():
    """Test full conversion and sorting."""
    # 1239.84198 / 800 ≈ 1.55 eV, 1239.84198 / 400 ≈ 3.10 eV
    wl = np.array([400.0, 800.0])
    data = np.array([40.0, 80.0])

    e_sorted, d_sorted = convert_and_sort(wl, data)
    # Wavelength 800nm (smaller energy) comes first after sorting by energy ascending
    assert e_sorted[0] < e_sorted[1]
    assert np.allclose(e_sorted[0], 1239.84198 / 800.0)
    assert np.allclose(d_sorted[0], 80.0)


def test_reflectance_to_kubelka_munk_valid():
    """Test Kubelka Munk calculations for typical inputs."""
    # R = 0.5 -> F(R) = (1 - 0.5)^2 / (2 * 0.5) = 0.25 / 1.0 = 0.25
    r = np.array([0.5])
    assert np.allclose(reflectance_to_kubelka_munk(r), 0.25)


def test_reflectance_to_kubelka_munk_boundary_values(caplog):
    """Test R=0 and R=1 boundary limits without crash and assert warnings."""
    # Test R=1
    r_one = np.array([1.0])
    assert np.allclose(reflectance_to_kubelka_munk(r_one), 0.0)

    # Test R=0 (should be clipped to 1e-10 and not divide by zero)
    r_zero = np.array([0.0])
    res_zero = reflectance_to_kubelka_munk(r_zero)
    assert np.isfinite(res_zero[0])
    assert res_zero[0] > 0.0

    # Verify no warning is logged for 1 point because fraction is 1/1 = 100% > 1%,
    # Wait, 1/1 = 100% which is indeed > 1%, so it should trigger a warning!
    # Let's check caplog
    with caplog.at_level(logging.WARNING):
        reflectance_to_kubelka_munk(np.array([0.0]))
        assert any("Clipping warning" in record.message for record in caplog.records)


def test_reflectance_to_kubelka_munk_no_warning_on_small_clip(caplog):
    """Test that warning is not logged when less than 1% of values are clipped."""
    # Create 200 points (1 clipped is 0.5% < 1%)
    r = np.ones(200) * 0.5
    r[0] = 0.0  # One out of bounds

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        reflectance_to_kubelka_munk(r)
        # Should not find warning
        assert not any("Clipping warning" in record.message for record in caplog.records)


def test_reflectance_to_kubelka_munk_warning_on_large_clip(caplog):
    """Test that warning is logged when more than 1% of values are clipped."""
    # Create 100 points, 2 clipped is 2.0% > 1%
    r = np.ones(100) * 0.5
    r[0] = 0.0
    r[1] = 1.1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        reflectance_to_kubelka_munk(r)
        assert any("Clipping warning" in record.message for record in caplog.records)
