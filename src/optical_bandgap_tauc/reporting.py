"""Reporting and export utilities.

This module provides functions to display analysis results in formatted terminal tables
using `rich`, and write output to CSV and JSON formats.
"""

import csv
import json
import logging
import os
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


def print_results_table(results: list[dict], quiet: bool = False) -> None:
    """Print results list in a clean terminal table using Rich.

    Highlights transitions with R^2 < 0.98 or recommended flags.

    Parameters
    ----------
    results : list[dict]
        List of result dictionaries. Each dict should contain keys:
        'sample_name', 'transition_type', 'band_gap_ev', 'r_squared',
        'window_start', 'window_end', 'method', 'is_recommended'.
    quiet : bool, optional
        If True, only outputs a final summary, suppressing debug table prints.
        Default is False.
    """
    if quiet or not results:
        return

    console = Console()
    table = Table(
        title="[bold cyan]Optical Band Gap Results Summary[/bold cyan]",
        title_justify="left",
    )

    table.add_column("Sample", style="dim", width=20)
    table.add_column("Transition Type", width=22)
    table.add_column("Eg (eV)", justify="right", style="bold green")
    table.add_column("R²", justify="right")
    table.add_column("Fit Window (eV)", justify="center")
    table.add_column("Method", style="dim")
    table.add_column("Status", justify="center")

    for res in results:
        r2 = res["r_squared"]
        r2_str = f"{r2:.4f}"
        status_str = ""

        # Highlight low R2
        if r2 < 0.98:
            r2_style = "[bold red]" + r2_str + "[/bold red]"
            status_str += "[bold red]⚠ Weak Fit[/bold red]"
        else:
            r2_style = f"{r2_str}"

        # Mark recommended
        if res.get("is_recommended", False):
            sample_style = f"[bold yellow]{res['sample_name']}[/bold yellow]"
            trans_style = (
                f"[bold yellow]{res['transition_type']}*[/bold yellow]"
            )
            if status_str:
                status_str += " | [bold yellow]★ Rec[/bold yellow]"
            else:
                status_str = "[bold yellow]★ Recommended[/bold yellow]"
        else:
            sample_style = res["sample_name"]
            trans_style = res["transition_type"]

        window_str = f"{res['window_start']:.3f} - {res['window_end']:.3f}"

        table.add_row(
            sample_style,
            trans_style,
            f"{res['band_gap_ev']:.3f}",
            r2_style,
            window_str,
            res["method"],
            status_str,
        )

    console.print("\n")
    console.print(table)
    console.print("[dim]* denotes recommended transition type[/dim]")
    console.print("\n")


def export_to_csv(filepath: str, results: list[dict]) -> None:
    """Export results to a CSV file.

    Parameters
    ----------
    filepath : str
        Target file path.
    results : list[dict]
        List of results dictionaries.
    """
    if not results:
        logger.debug("No results to write to CSV.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    headers = [
        "sample_name",
        "transition_type",
        "band_gap_ev",
        "r_squared",
        "window_start",
        "window_end",
        "method",
        "is_recommended",
        "cross_band_gap_ev",
        "cross_r_squared",
        "disagreement_ev",
    ]

    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for res in results:
                # Filter dictionary keys to match CSV headers
                row = {h: res.get(h, "") for h in headers}
                # Standardize boolean print
                row["is_recommended"] = str(row["is_recommended"]).lower()
                writer.writerow(row)
        logger.debug(f"Successfully exported CSV results to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export CSV results to {filepath}: {e}")
        raise


def export_to_json(filepath: str, results: list[dict]) -> None:
    """Export results to a JSON file.

    Parameters
    ----------
    filepath : str
        Target file path.
    results : list[dict]
        List of results dictionaries.
    """
    if not results:
        logger.debug("No results to write to JSON.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.debug(f"Successfully exported JSON results to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export JSON results to {filepath}: {e}")
        raise
