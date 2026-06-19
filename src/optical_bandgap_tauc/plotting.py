"""Plotting functions for Tauc analysis results.

This module provides functions to generate high-quality research-grade Tauc plots
and batch overlay plots.
"""

import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from optical_bandgap_tauc.edge_detection import LinearRegimeResult

# Use non-interactive backend for server/CLI use cases
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Premium, research-grade styling constants
DATA_COLOR = "#2B5C8F"  # Deep blue
SMOOTH_COLOR = "#008080"  # Teal
FIT_COLOR = "#E64A19"  # Vibrant orange-red
WINDOW_COLOR = "#FFF59D"  # Soft yellow shading


def plot_single_tauc(
    energy: np.ndarray,
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    result: LinearRegimeResult,
    transition_type: str,
    sample_name: str,
    output_dir: str,
    formats: list[str] = ["png"],
) -> None:
    """Generate and save a publication-quality Tauc plot for a single sample.

    Shows raw data, smoothed curve, highlighted fit window, and linear extrapolation.

    Parameters
    ----------
    energy : np.ndarray
        Array of photon energy values (eV).
    y_raw : np.ndarray
        Array of raw Tauc variable values (F(R)*E)^n.
    y_smooth : np.ndarray
        Array of smoothed Tauc variable values.
    result : LinearRegimeResult
        Result from linear-regime detection.
    transition_type : str
        Transition type key (e.g., 'direct-allowed').
    sample_name : str
        Name of the sample.
    output_dir : str
        Directory to save the plots.
    formats : list[str], optional
        Output format extensions (e.g. ['png', 'pdf']). Default is ['png'].
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

    # 1. Plot raw and smoothed data
    ax.plot(energy, y_raw, color=DATA_COLOR, alpha=0.3, label="Raw Data", lw=1)
    ax.plot(energy, y_smooth, color=SMOOTH_COLOR, label="SG Smoothed", lw=1.8)

    # 2. Highlight fit window
    ax.axvspan(
        result.window_start,
        result.window_end,
        color=WINDOW_COLOR,
        alpha=0.35,
        label=f"Fit Window ({result.window_start:.3f}-{result.window_end:.3f} eV)",
    )

    # 3. Extrapolate fit line to the x-axis (y = 0)
    # Fit line: y = m*E + c. Intercept with y=0 is Eg.
    m = result.slope
    c = result.intercept
    Eg = result.band_gap_ev

    # Determine x-range for the fit line. It should run from Eg to at least the end of the window.
    x_min_fit = min(Eg, result.window_start) - 0.05
    x_max_fit = max(Eg, result.window_end) + 0.1

    # Ensure range doesn't go negative or look ridiculous
    if Eg > 0:
        x_min_fit = max(x_min_fit, Eg - 0.2)
    else:
        x_min_fit = max(x_min_fit, 0.0)

    x_fit_line = np.linspace(x_min_fit, x_max_fit, 100)
    y_fit_line = m * x_fit_line + c

    # Clip fit line to not go below 0 for visual neatness
    y_fit_line_clipped = np.maximum(0.0, y_fit_line)

    ax.plot(
        x_fit_line,
        y_fit_line_clipped,
        color=FIT_COLOR,
        linestyle="--",
        lw=2.0,
        label="Linear Extrapolation",
    )

    # 4. Highlight the Eg point on the axis
    ax.plot(Eg, 0, marker="o", color=FIT_COLOR, markersize=8, zorder=5)

    # Annotate Eg on the plot
    y_limits = ax.get_ylim()
    y_span = y_limits[1] - y_limits[0]

    # Draw annotation textbox
    transition_pretty = transition_type.replace("-", " ").title()
    textstr = "\n".join(
        (
            f"Sample: {sample_name}",
            f"Transition: {transition_pretty}",
            rf"$E_g$ = {Eg:.3f} eV",
            rf"$R^2$ = {result.r_squared:.4f}",
            f"Method: {result.method}",
        )
    )

    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="silver")
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    # X-axis annotation at Eg
    ax.annotate(
        f"{Eg:.3f} eV",
        xy=(Eg, 0),
        xytext=(Eg + 0.05, y_span * 0.05),
        arrowprops=dict(
            arrowstyle="->", color=FIT_COLOR, connectionstyle="arc3,rad=.2"
        ),
        fontsize=10,
        color=FIT_COLOR,
        weight="bold",
    )

    # 5. Graph labels and aesthetics
    # The standard Tauc variable label contains (F(R)*E)^n or (alpha*h*nu)^n
    n_labels = {
        "direct-allowed": r"$(F(R) \cdot E)^2$",
        "indirect-allowed": r"$(F(R) \cdot E)^{1/2}$",
        "direct-forbidden": r"$(F(R) \cdot E)^{2/3}$",
        "indirect-forbidden": r"$(F(R) \cdot E)^{1/3}$",
    }
    y_label = n_labels.get(transition_type, r"$(F(R) \cdot E)^n$")

    ax.set_xlabel("Photon Energy $E$ (eV)", fontsize=12, fontweight="medium")
    ax.set_ylabel(y_label, fontsize=12, fontweight="medium")
    ax.set_title(
        f"Tauc Plot - {sample_name} ({transition_pretty})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="silver")

    # Set appropriate limits
    ax.set_xlim(
        max(0.0, np.min(energy) - 0.1),
        np.max(energy) + 0.1,
    )
    # Ensure y-axis doesn't go excessively negative
    ax.set_ylim(-y_span * 0.02, y_limits[1])

    # Tight layout and save in all requested formats
    plt.tight_layout()
    safe_sample_name = "".join(
        [c if c.isalnum() or c in ("-", "_") else "_" for c in sample_name]
    )

    for fmt in formats:
        fmt = fmt.strip().lower().replace(".", "")
        filename = f"{safe_sample_name}_{transition_type}_tauc.{fmt}"
        dest_path = os.path.join(output_dir, filename)
        fig.savefig(dest_path, format=fmt, bbox_inches="tight")
        logger.debug(f"Saved single Tauc plot: {dest_path}")

    plt.close(fig)


def plot_batch_overlay(
    samples_data: list[dict],
    transition_type: str,
    output_dir: str,
    formats: list[str] = ["png"],
) -> None:
    """Generate and save a comparison overlay plot of multiple Tauc curves.

    Parameters
    ----------
    samples_data : list[dict]
        A list of dictionaries. Each dictionary must contain:
        - 'sample_name': str
        - 'energy': np.ndarray
        - 'y_smooth': np.ndarray (smoothed Tauc values)
        - 'band_gap_ev': float
        - 'r_squared': float
    transition_type : str
        Transition type key (e.g. 'direct-allowed').
    output_dir : str
        Directory to save the plots.
    formats : list[str], optional
        Output format extensions. Default is ['png'].
    """
    if not samples_data:
        logger.debug("No sample data provided for batch overlay plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Use a standard colormap to make lines look premium and distinct
    cmap = matplotlib.colormaps["tab10"]

    for i, sdata in enumerate(samples_data):
        color = cmap(i % 10)
        name = sdata["sample_name"]
        energy = sdata["energy"]
        y = sdata["y_smooth"]
        Eg = sdata["band_gap_ev"]

        # Plot curve
        ax.plot(
            energy,
            y,
            color=color,
            lw=1.5,
            label=f"{name} ($E_g$={Eg:.2f} eV)",
        )

        # Plot Eg marker
        ax.plot(Eg, 0, marker="o", color=color, markersize=6)

    transition_pretty = transition_type.replace("-", " ").title()
    n_labels = {
        "direct-allowed": r"$(F(R) \cdot E)^2$",
        "indirect-allowed": r"$(F(R) \cdot E)^{1/2}$",
        "direct-forbidden": r"$(F(R) \cdot E)^{2/3}$",
        "indirect-forbidden": r"$(F(R) \cdot E)^{1/3}$",
    }
    y_label = n_labels.get(transition_type, r"$(F(R) \cdot E)^n$")

    ax.set_xlabel("Photon Energy $E$ (eV)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"Batch Overlay Tauc Comparison ({transition_pretty})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    # Limit legend size if there are too many samples
    ax.legend(
        loc="upper left",
        fontsize=9,
        ncol=2 if len(samples_data) > 5 else 1,
        framealpha=0.9,
    )

    # Standardize plot limits
    all_energy = np.concatenate([s["energy"] for s in samples_data])
    ax.set_xlim(
        max(0.0, np.min(all_energy) - 0.1),
        np.max(all_energy) + 0.1,
    )

    y_limits = ax.get_ylim()
    ax.set_ylim(-0.02 * (y_limits[1] - y_limits[0]), y_limits[1])

    plt.tight_layout()

    for fmt in formats:
        fmt = fmt.strip().lower().replace(".", "")
        filename = f"batch_overlay_{transition_type}_tauc.{fmt}"
        dest_path = os.path.join(output_dir, filename)
        fig.savefig(dest_path, format=fmt, bbox_inches="tight")
        logger.debug(f"Saved batch overlay plot: {dest_path}")

    plt.close(fig)
