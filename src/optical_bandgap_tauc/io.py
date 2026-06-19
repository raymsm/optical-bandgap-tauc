"""Input/Output functions for reading DRS data files.

This module handles parsing of raw data files from different instruments,
including auto-detecting delimiters, headers, and skipping metadata.
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def parse_drs_file(
    filepath: str, instrument_format: str = "generic"
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a DRS data file to extract wavelength and reflectance/absorbance.

    Auto-detects delimiters (comma, tab, semicolon, whitespace), header presence,
    skips instrument metadata headers and footers, and handles decimal commas.

    Parameters
    ----------
    filepath : str
        Path to the raw spectrum file.
    instrument_format : str, optional
        Format style to prioritize. One of {'generic', 'shimadzu', 'perkinelmer'}.
        Default is 'generic'.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (wavelength_nm, raw_data).

    Raises
    ------
    FileNotFoundError
        If the filepath does not exist.
    ValueError
        If no numeric data could be extracted or formatting is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Spectrum file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    detected_delim = None
    data_start_idx = None
    wavelengths = []
    data_values = []

    # 1. Scan for the first line containing at least two valid numeric values
    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue

        # Try common delimiters in order of likelihood
        # Semicolon is tested for European files with decimal commas
        for delim in [",", "\t", ";", None]:
            parts = line_str.split(delim) if delim is not None else line_str.split()
            if len(parts) >= 2:
                try:
                    # Clean the values (handle decimal commas if not comma-delimited)
                    p0 = parts[0].strip()
                    p1 = parts[1].strip()
                    if delim != ",":
                        p0 = p0.replace(",", ".")
                        p1 = p1.replace(",", ".")

                    float(p0)
                    float(p1)
                    # If conversion succeeds, we found the data start and delimiter!
                    detected_delim = delim
                    data_start_idx = line_idx
                    break
                except ValueError:
                    pass
        if data_start_idx is not None:
            break

    if data_start_idx is None:
        raise ValueError(
            f"Could not locate numeric data columns in the file: {filepath}"
        )

    # Log details about auto-detection
    delim_name = (
        detected_delim if detected_delim is not None else "whitespace/tab"
    )
    logger.debug(
        f"Auto-detected delimiter: '{delim_name}', starting on line {data_start_idx+1}"
    )

    # 2. Check if a header line is present immediately before the data
    header_line = None
    if data_start_idx > 0:
        # Trace back to find the closest non-empty line before data
        for idx in range(data_start_idx - 1, -1, -1):
            temp = lines[idx].strip()
            if temp and not temp.startswith("#"):
                header_line = temp
                break

    if header_line:
        logger.debug(f"Detected potential header line: '{header_line}'")

    # 3. Parse all subsequent lines
    for line_idx in range(data_start_idx, len(lines)):
        line_str = lines[line_idx].strip()
        if not line_str or line_str.startswith("#"):
            continue

        parts = (
            line_str.split(detected_delim)
            if detected_delim is not None
            else line_str.split()
        )
        if len(parts) < 2:
            # Skip rows that don't match the column counts (e.g. metadata footers)
            continue

        try:
            p0 = parts[0].strip()
            p1 = parts[1].strip()
            if detected_delim != ",":
                p0 = p0.replace(",", ".")
                p1 = p1.replace(",", ".")

            wl = float(p0)
            val = float(p1)
            wavelengths.append(wl)
            data_values.append(val)
        except ValueError:
            # Skip invalid rows (e.g., text warnings/footers)
            continue

    if len(wavelengths) < 2:
        raise ValueError(
            f"File '{filepath}' contains fewer than 2 valid data points."
        )

    return np.array(wavelengths, dtype=float), np.array(data_values, dtype=float)
