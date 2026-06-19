"""Integration tests for the command-line interface (CLI)."""

import json
import os
from pathlib import Path
import numpy as np
import pytest
from typer.testing import CliRunner

from optical_bandgap_tauc.cli import app

runner = CliRunner()


def create_mock_drs_file(
    filepath: Path,
    delimiter: str = ",",
    header: bool = True,
    decimal_comma: bool = False,
) -> None:
    """Helper to generate a mock DRS file with a clear reflection edge."""
    # 100 points, wavelength from 300nm to 800nm
    wls = np.linspace(300.0, 800.0, 100)
    # Create reflectance ramp that goes from 10% to 90% (midpoint at 500nm)
    rs = 10.0 + 80.0 / (1.0 + np.exp(-(wls - 500.0) / 30.0))

    with open(filepath, "w", encoding="utf-8") as f:
        if header:
            f.write(f"Wavelength (nm){delimiter}Reflectance (%R)\n")
        for w, r in zip(wls, rs):
            w_str = f"{w:.3f}"
            r_str = f"{r:.3f}"
            if decimal_comma:
                w_str = w_str.replace(".", ",")
                r_str = r_str.replace(".", ",")
            f.write(f"{w_str}{delimiter}{r_str}\n")


def test_cli_analyze_flow(tmp_path: Path):
    """Test standard single file analysis flow and check outputs."""
    input_file = tmp_path / "sample_A.csv"
    create_mock_drs_file(input_file, delimiter=",", header=True)

    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(input_file),
            "--output-dir",
            str(out_dir),
            "--format",
            "png",
            "--export",
            "csv",
            "--export",
            "json",
        ],
    )

    assert result.exit_code == 0

    # Check generated files
    assert os.path.exists(out_dir / "sample_A_results.csv")
    assert os.path.exists(out_dir / "sample_A_results.json")
    assert os.path.exists(out_dir / "sample_A_direct-allowed_tauc.png")

    # Read and inspect JSON results
    with open(out_dir / "sample_A_results.json", "r") as json_f:
        data = json.load(json_f)

    # Ensure all four transition types are analyzed and recommendation is set
    assert len(data) == 4
    has_rec = any(item["is_recommended"] for item in data)
    assert has_rec

    # Check that band gap is a valid positive float
    for item in data:
        assert item["band_gap_ev"] > 0.0
        assert item["r_squared"] > 0.0
        assert item["window_start"] < item["window_end"]


def test_cli_batch_flow(tmp_path: Path):
    """Test batch processing of a directory containing multiple files."""
    input_dir = tmp_path / "raw_data"
    input_dir.mkdir()

    # Create 3 mock DRS files
    for name in ["sample_1.txt", "sample_2.txt", "sample_3.txt"]:
        create_mock_drs_file(input_dir / name, delimiter="\t", header=True)

    out_dir = tmp_path / "batch_results"

    result = runner.invoke(
        app,
        [
            "batch",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--transition",
            "direct-allowed",
            "--format",
            "png",
            "--export",
            "csv",
        ],
    )

    assert result.exit_code == 0

    # Check individual outputs are created
    assert os.path.exists(out_dir / "sample_1_results.csv")
    assert os.path.exists(out_dir / "sample_2_results.csv")
    assert os.path.exists(out_dir / "sample_3_results.csv")
    assert os.path.exists(out_dir / "sample_1_direct-allowed_tauc.png")

    # Check batch overlay plot and comparison tables are created
    assert os.path.exists(out_dir / "batch_overlay_direct-allowed_tauc.png")
    assert os.path.exists(out_dir / "batch_combined_flat.csv")
    assert os.path.exists(out_dir / "batch_summary_comparison.csv")


def test_cli_delimiter_variants(tmp_path: Path):
    """Test that CSV format auto-detection works for multiple layout permutations."""
    # Test European standard style: Semicolon delimiter with decimal comma
    euro_file = tmp_path / "euro_style.csv"
    create_mock_drs_file(euro_file, delimiter=";", header=True, decimal_comma=True)

    out_dir = tmp_path / "euro_out"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(euro_file),
            "--output-dir",
            str(out_dir),
            "--no-plot",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(out_dir / "euro_style_results.json")


def test_cli_manual_window_override(tmp_path: Path):
    """Test analyze execution with a manual window width override."""
    input_file = tmp_path / "sample_override.csv"
    create_mock_drs_file(input_file, delimiter=",", header=True)

    out_dir = tmp_path / "override_out"

    # Request a fixed 0.25 eV fit window
    result = runner.invoke(
        app,
        [
            "analyze",
            str(input_file),
            "--output-dir",
            str(out_dir),
            "--edge-window-ev",
            "0.25",
            "--no-plot",
        ],
    )
    assert result.exit_code == 0

    with open(out_dir / "sample_override_results.json", "r") as json_f:
        data = json.load(json_f)

    for item in data:
        assert item["method"] == "manual-override"
        # The window span should be close to 0.25 eV
        window_span = item["window_end"] - item["window_start"]
        assert np.isclose(window_span, 0.25, atol=0.05)
