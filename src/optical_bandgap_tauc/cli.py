"""Command-line interface (CLI) for the optical band gap analysis tool.

This module implements the typer CLI, providing the subcommands:
- `analyze` for a single spectrum file.
- `batch` for processing multiple spectra in a directory.
"""

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import typer

from optical_bandgap_tauc.edge_detection import detect_linear_regime
from optical_bandgap_tauc.io import parse_drs_file
from optical_bandgap_tauc.fitting import fit_linear_regression
from optical_bandgap_tauc.kubelka_munk import (
    convert_and_sort,
    reflectance_to_kubelka_munk,
)
from optical_bandgap_tauc.plotting import plot_batch_overlay, plot_single_tauc
from optical_bandgap_tauc.reporting import (
    export_to_csv,
    export_to_json,
    print_results_table,
)
from optical_bandgap_tauc.tauc import (
    TRANSITIONS,
    compute_tauc_variable,
    smooth_tauc_data,
)

app = typer.Typer(
    help="Automated DRS data processing and Tauc plot optical bandgap extraction.",
    add_completion=False,
)

logger = logging.getLogger("optical_bandgap_tauc")


def configure_logging(verbose: bool) -> None:
    """Configure python logging level and format.

    Parameters
    ----------
    verbose : bool
        If True, sets level to DEBUG, otherwise INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(log_level)


def run_pipeline_for_sample(
    filepath: Path,
    input_type: str,
    transition_filter: str,
    smooth_window: Optional[int],
    smooth_order: Optional[int],
    edge_window_ev: Optional[float],
    disagreement_warn_ev: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Run the core science pipeline on a single spectrum file.

    Parameters
    ----------
    filepath : Path
        Path to raw spectroscopy file.
    input_type : str
        Type of input data column: reflectance-pct, reflectance-frac, or absorbance.
    transition_filter : str
        Transition filter, one of the keys of TRANSITIONS, or 'all'.
    smooth_window : int | None
        Override for S-G window length.
    smooth_order : int | None
        Override for S-G polynomial order.
    edge_window_ev : float | None
        Override for fit window width.
    disagreement_warn_ev : float, optional
        Threshold for method disagreement warning. Default is 0.05.

    Returns
    -------
    tuple
        (energy_sorted, f_r, list_of_transition_results_dicts, raw_tauc_curves_dict)
    """
    sample_name = filepath.stem
    logger.info(f"Processing sample: {sample_name} ({filepath.name})")

    # 1. Parse file
    wavelength, raw_data = parse_drs_file(str(filepath))

    # 2. Convert and sort
    energy, data = convert_and_sort(wavelength, raw_data)

    # 3. Kubelka-Munk proxy conversion
    if input_type == "reflectance-pct":
        f_r = reflectance_to_kubelka_munk(data / 100.0)
    elif input_type == "reflectance-frac":
        f_r = reflectance_to_kubelka_munk(data)
    elif input_type == "absorbance":
        f_r = data
    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # 4. Resolve transition types
    if transition_filter == "all":
        transitions_to_run = list(TRANSITIONS.keys())
    elif transition_filter in TRANSITIONS:
        transitions_to_run = [transition_filter]
    else:
        raise ValueError(
            f"Invalid transition type: {transition_filter}. "
            f"Select from: all, {', '.join(TRANSITIONS.keys())}"
        )

    results = []
    raw_curves = {}

    # 5. Process each transition type
    for trans in transitions_to_run:
        logger.debug(f"Analyzing transition: {trans} for sample {sample_name}")
        y_raw = compute_tauc_variable(energy, f_r, trans)
        raw_curves[trans] = y_raw

        # Run linear-regime detection (includes S-G smoothing internally)
        fit_res = detect_linear_regime(
            energy,
            y_raw,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
            edge_window_ev=edge_window_ev,
            disagreement_warn_ev=disagreement_warn_ev,
        )

        res_dict = {
            "sample_name": sample_name,
            "transition_type": trans,
            "band_gap_ev": fit_res.band_gap_ev,
            "r_squared": fit_res.r_squared,
            "window_start": fit_res.window_start,
            "window_end": fit_res.window_end,
            "method": fit_res.method,
            "slope": fit_res.slope,
            "intercept": fit_res.intercept,
            "cross_band_gap_ev": fit_res.cross_band_gap_ev,
            "cross_r_squared": fit_res.cross_r_squared,
            "disagreement_ev": fit_res.disagreement_ev,
            "is_recommended": False,  # Will be set later
        }
        results.append(res_dict)

    # 6. Recommendation: pick the transition with the highest R^2 that has low disagreement
    # If none, just pick the highest R^2
    best_idx = -1
    best_r2 = -1.0

    # First pass: try to find highest R^2 with disagreement <= disagreement_warn_ev
    for idx, r in enumerate(results):
        r2 = r["r_squared"]
        dis = r["disagreement_ev"]
        # If disagreement is None (e.g. fit failed or couldn't run), ignore it
        if dis is not None and dis <= disagreement_warn_ev:
            if r2 > best_r2:
                best_r2 = r2
                best_idx = idx

    # Second pass: if no transition met the disagreement limit, just pick highest R^2
    if best_idx == -1:
        for idx, r in enumerate(results):
            r2 = r["r_squared"]
            if r2 > best_r2:
                best_r2 = r2
                best_idx = idx

    if best_idx != -1:
        results[best_idx]["is_recommended"] = True
        logger.info(
            f"Recommended transition for {sample_name}: "
            f"{results[best_idx]['transition_type']} (Eg={results[best_idx]['band_gap_ev']:.3f} eV, R²={results[best_idx]['r_squared']:.4f})"
        )

    return energy, f_r, results, raw_curves


@app.command("analyze")
def analyze(
    input_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to raw DRS spectroscopy CSV/txt data file.",
    ),
    input_type: str = typer.Option(
        "reflectance-pct",
        "--input-type",
        "-t",
        help="Input column format: 'reflectance-pct' (0-100% R), 'reflectance-frac' (0.0-1.0 R), or 'absorbance'.",
    ),
    transition: str = typer.Option(
        "all",
        "--transition",
        "-tr",
        help="Filter transition type: direct-allowed, indirect-allowed, direct-forbidden, indirect-forbidden, or all.",
    ),
    smooth_window: Optional[int] = typer.Option(
        None,
        "--smooth-window",
        help="Savitzky-Golay filter window length (must be odd).",
    ),
    smooth_order: Optional[int] = typer.Option(
        None,
        "--smooth-order",
        help="Savitzky-Golay polynomial order (default: 3).",
    ),
    edge_window_ev: Optional[float] = typer.Option(
        None,
        "--edge-window-ev",
        help="Manual override of fit window width in eV (skips auto edge detection).",
    ),
    output_dir: Path = typer.Option(
        Path("./bandgap_results/"),
        "--output-dir",
        "-o",
        help="Directory to save output plots and tables.",
    ),
    format: List[str] = typer.Option(
        ["png"],
        "--format",
        "-f",
        help="Output plot file format (png, pdf, svg). Can be specified multiple times.",
    ),
    export: List[str] = typer.Option(
        ["csv", "json"],
        "--export",
        "-e",
        help="Numeric results file format (csv, json). Can be specified multiple times.",
    ),
    no_plot: bool = typer.Option(
        False,
        "--no-plot",
        help="Suppress graphical plot generation (useful for server/headless run).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose DEBUG level log messages.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress print of the results table in terminal.",
    ),
) -> None:
    """Analyze a single spectroscopy file to extract optical band gaps."""
    configure_logging(verbose)
    try:
        energy, f_r, results, raw_curves = run_pipeline_for_sample(
            filepath=input_file,
            input_type=input_type,
            transition_filter=transition,
            smooth_window=smooth_window,
            smooth_order=smooth_order,
            edge_window_ev=edge_window_ev,
        )

        sample_name = input_file.stem
        os.makedirs(output_dir, exist_ok=True)

        # 1. Print terminal table
        print_results_table(results, quiet=quiet)

        # 2. Export numeric files
        safe_sample_name = "".join(
            [c if c.isalnum() or c in ("-", "_") else "_" for c in sample_name]
        )
        if "csv" in export:
            csv_path = os.path.join(output_dir, f"{safe_sample_name}_results.csv")
            export_to_csv(csv_path, results)
            logger.info(f"Results CSV written to: {csv_path}")

        if "json" in export:
            json_path = os.path.join(output_dir, f"{safe_sample_name}_results.json")
            export_to_json(json_path, results)
            logger.info(f"Results JSON written to: {json_path}")

        # 3. Generate plots
        if not no_plot:
            for r in results:
                t_type = r["transition_type"]
                y_raw = raw_curves[t_type]
                y_smooth = smooth_tauc_data(
                    y_raw, user_window=smooth_window, user_order=smooth_order
                )

                # Reconstruct LinearRegimeResult for plotting
                result_obj = detect_linear_regime(
                    energy,
                    y_raw,
                    smooth_window=smooth_window,
                    smooth_order=smooth_order,
                    edge_window_ev=edge_window_ev,
                )

                plot_single_tauc(
                    energy=energy,
                    y_raw=y_raw,
                    y_smooth=y_smooth,
                    result=result_obj,
                    transition_type=t_type,
                    sample_name=sample_name,
                    output_dir=str(output_dir),
                    formats=format,
                )

    except Exception as e:
        logger.error(f"Error occurred during analysis: {e}", exc_info=verbose)
        raise typer.Exit(code=1)


@app.command("batch")
def batch(
    directory: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing spectrum data files.",
    ),
    input_type: str = typer.Option(
        "reflectance-pct",
        "--input-type",
        "-t",
        help="Input column format: 'reflectance-pct' (0-100% R), 'reflectance-frac' (0.0-1.0 R), or 'absorbance'.",
    ),
    transition: str = typer.Option(
        "all",
        "--transition",
        "-tr",
        help="Filter transition type: direct-allowed, indirect-allowed, direct-forbidden, indirect-forbidden, or all.",
    ),
    smooth_window: Optional[int] = typer.Option(
        None,
        "--smooth-window",
        help="Savitzky-Golay filter window length (must be odd).",
    ),
    smooth_order: Optional[int] = typer.Option(
        None,
        "--smooth-order",
        help="Savitzky-Golay polynomial order (default: 3).",
    ),
    edge_window_ev: Optional[float] = typer.Option(
        None,
        "--edge-window-ev",
        help="Manual override of fit window width in eV.",
    ),
    output_dir: Path = typer.Option(
        Path("./bandgap_results/"),
        "--output-dir",
        "-o",
        help="Directory to save output plots and tables.",
    ),
    format: List[str] = typer.Option(
        ["png"],
        "--format",
        "-f",
        help="Output plot file format (png, pdf, svg). Can be specified multiple times.",
    ),
    export: List[str] = typer.Option(
        ["csv", "json"],
        "--export",
        "-e",
        help="Numeric results file format (csv, json). Can be specified multiple times.",
    ),
    no_plot: bool = typer.Option(
        False,
        "--no-plot",
        help="Suppress graphical plot generation (useful for server/headless run).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose DEBUG level log messages.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress print of individual results tables in terminal.",
    ),
) -> None:
    """Batch processes a directory of DRS files.

    Performs full Tauc analysis on each, and outputs summary tables plus comparison plots.
    """
    configure_logging(verbose)
    try:
        # Find all files in directory
        # Filter files to check common extension types
        extensions = [".csv", ".txt", ".dat", ".tsv"]
        files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]

        if not files:
            logger.warning(
                f"No files matching extensions {extensions} found in directory {directory}"
            )
            raise typer.Exit(code=0)

        logger.info(f"Found {len(files)} files to analyze in {directory}")

        all_sample_results = []
        batch_plot_data = {}  # transition_type -> list of plot dictionaries

        for f in files:
            try:
                energy, f_r, results, raw_curves = run_pipeline_for_sample(
                    filepath=f,
                    input_type=input_type,
                    transition_filter=transition,
                    smooth_window=smooth_window,
                    smooth_order=smooth_order,
                    edge_window_ev=edge_window_ev,
                )

                all_sample_results.extend(results)

                # Generate individual output files
                sample_name = f.stem
                safe_name = "".join(
                    [
                        c
                        if c.isalnum() or c in ("-", "_")
                        else "_"
                        for c in sample_name
                    ]
                )

                if "csv" in export:
                    csv_path = os.path.join(
                        output_dir, f"{safe_name}_results.csv"
                    )
                    export_to_csv(csv_path, results)

                if "json" in export:
                    json_path = os.path.join(
                        output_dir, f"{safe_name}_results.json"
                    )
                    export_to_json(json_path, results)

                for r in results:
                    t_type = r["transition_type"]
                    y_raw = raw_curves[t_type]
                    y_smooth = smooth_tauc_data(
                        y_raw,
                        user_window=smooth_window,
                        user_order=smooth_order,
                    )

                    # Accumulate for overlay plots
                    if t_type not in batch_plot_data:
                        batch_plot_data[t_type] = []
                    batch_plot_data[t_type].append(
                        {
                            "sample_name": sample_name,
                            "energy": energy,
                            "y_smooth": y_smooth,
                            "band_gap_ev": r["band_gap_ev"],
                            "r_squared": r["r_squared"],
                        }
                    )

                    if not no_plot:
                        # Reconstruct LinearRegimeResult for plotting
                        result_obj = detect_linear_regime(
                            energy,
                            y_raw,
                            smooth_window=smooth_window,
                            smooth_order=smooth_order,
                            edge_window_ev=edge_window_ev,
                        )
                        plot_single_tauc(
                            energy=energy,
                            y_raw=y_raw,
                            y_smooth=y_smooth,
                            result=result_obj,
                            transition_type=t_type,
                            sample_name=sample_name,
                            output_dir=str(output_dir),
                            formats=format,
                        )

            except Exception as ex:
                logger.error(
                    f"Skipping file {f.name} due to processing error: {ex}",
                    exc_info=verbose,
                )

        if not all_sample_results:
            logger.error("No spectra were successfully processed in batch mode.")
            raise typer.Exit(code=1)

        # Print terminal table for the entire batch
        print_results_table(all_sample_results, quiet=quiet)

        # Export combined tables
        # 1. Full flat CSV of all samples and transitions
        combined_flat_csv = os.path.join(output_dir, "batch_combined_flat.csv")
        export_to_csv(combined_flat_csv, all_sample_results)
        logger.info(f"Combined flat results CSV written to: {combined_flat_csv}")

        # 2. Summary pivot CSV (ranked by recommended transition R2 descending)
        # Create columns: sample_name, recommended_transition, recommended_Eg_ev, recommended_R2,
        # plus Eg for each transition type
        summary_records = {}
        for r in all_sample_results:
            sname = r["sample_name"]
            if sname not in summary_records:
                summary_records[sname] = {
                    "sample_name": sname,
                    "recommended_transition": "",
                    "recommended_Eg_ev": "",
                    "recommended_R2": "",
                    "direct_allowed_Eg": "",
                    "direct_allowed_R2": "",
                    "indirect_allowed_Eg": "",
                    "indirect_allowed_R2": "",
                    "direct_forbidden_Eg": "",
                    "direct_forbidden_R2": "",
                    "indirect_forbidden_Eg": "",
                    "indirect_forbidden_R2": "",
                }

            t_type = r["transition_type"]
            eg_val = r["band_gap_ev"]
            r2_val = r["r_squared"]

            if t_type == "direct-allowed":
                summary_records[sname]["direct_allowed_Eg"] = eg_val
                summary_records[sname]["direct_allowed_R2"] = r2_val
            elif t_type == "indirect-allowed":
                summary_records[sname]["indirect_allowed_Eg"] = eg_val
                summary_records[sname]["indirect_allowed_R2"] = r2_val
            elif t_type == "direct-forbidden":
                summary_records[sname]["direct_forbidden_Eg"] = eg_val
                summary_records[sname]["direct_forbidden_R2"] = r2_val
            elif t_type == "indirect-forbidden":
                summary_records[sname]["indirect_forbidden_Eg"] = eg_val
                summary_records[sname]["indirect_forbidden_R2"] = r2_val

            if r.get("is_recommended", False):
                summary_records[sname]["recommended_transition"] = t_type
                summary_records[sname]["recommended_Eg_ev"] = eg_val
                summary_records[sname]["recommended_R2"] = r2_val

        # Sort samples by their recommended R^2 descending
        sorted_samples = sorted(
            list(summary_records.values()),
            key=lambda x: x["recommended_R2"]
            if x["recommended_R2"] != ""
            else -1.0,
            reverse=True,
        )

        # Write to summary csv
        summary_csv_path = os.path.join(output_dir, "batch_summary_comparison.csv")
        summary_headers = [
            "sample_name",
            "recommended_transition",
            "recommended_Eg_ev",
            "recommended_R2",
            "direct_allowed_Eg",
            "direct_allowed_R2",
            "indirect_allowed_Eg",
            "indirect_allowed_R2",
            "direct_forbidden_Eg",
            "direct_forbidden_R2",
            "indirect_forbidden_Eg",
            "indirect_forbidden_R2",
        ]
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_headers)
            writer.writeheader()
            writer.writerows(sorted_samples)
        logger.info(
            f"Combined summary comparison CSV written to: {summary_csv_path}"
        )

        # Generate batch overlay plots
        if not no_plot:
            for t_type, p_list in batch_plot_data.items():
                # Sort plot list by Eg or name for consistency
                p_list_sorted = sorted(p_list, key=lambda x: x["sample_name"])
                plot_batch_overlay(
                    samples_data=p_list_sorted,
                    transition_type=t_type,
                    output_dir=str(output_dir),
                    formats=format,
                )

    except Exception as e:
        logger.error(f"Error during batch execution: {e}", exc_info=verbose)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
