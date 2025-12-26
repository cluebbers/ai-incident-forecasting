import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecast_helpers import ForecastDiagnostics, ForecastResult


def _format_output_path(output_name: str) -> str:
    safe_name = output_name.strip() if output_name else "total_incidents.pdf"
    if not safe_name:
        safe_name = "total_incidents.pdf"
    if "." not in os.path.basename(safe_name):
        safe_name = f"{safe_name}.pdf"
    return os.path.join("../output", safe_name)

def plot_total_panel_subplots(
    res: ForecastResult,
    diagnostics: Optional[ForecastDiagnostics] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_uncertainty: bool = True,
    output_name: str = "total_incidents.pdf",
):
    """Enhanced total panel plot with clear uncertainty visualization"""
    fig, ax = plt.subplots(
        constrained_layout=True, figsize=figsize if figsize is not None else (6.25, 2.16)
    )

    # Multiple confidence levels
    alphas = [0.9]
    colors = ["#fee5d9", "#fcae91", "#fb6a4a"]

    # for alpha, color in zip(alphas, colors):
    #     lo = res.fore_total_lo
    #     hi = res.fore_total_hi
    #     ax.fill_between(
    #         res.years_fore,
    #         lo,
    #         hi,
    #         color=color,
    #         alpha=0.3,
    #         label=f"{int(100*alpha)}% PI",
    #     )
    if show_uncertainty:
        lo = res.fore_total_lo
        hi = res.fore_total_hi
        ax.fill_between(
            res.years_fore,
            lo,
            hi,
            alpha=0.3,
            label="90% PI",
        )

    actual_total = res.actual_by_year_full.sum(axis=1)

    ax.scatter(
        actual_total.index,
        actual_total.values,
        s=18,
        label="Total actual (incl. YTD est)",
    )

    if res.ytd_actual_total is not None and res.have_ytd and res.last_month is not None:
        ax.scatter(
            [res.ytd_year],
            [res.ytd_actual_total],
            s=28,
            marker="x",
            label=f"{res.ytd_year} YTD actual (m≤{res.last_month})",
        )

    ax.plot(
        res.fore_total.index,
        res.fore_total.values,
        "--",
        lw=2,
        label="Total forecast (median)",
    )
    # ax.set_title(
    #     f"Total incidents ({res.cat_col}) with {res.ytd_year} YTD assimilation"
    # )
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend()
    x_min_auto = min(actual_total.index.min(), res.years_fore.min())
    x_max_auto = max(actual_total.index.max(), res.years_fore.max())
    ax.set_xlim(x_min if x_min is not None else x_min_auto,
                x_max if x_max is not None else x_max_auto)
    y_min_auto = -50
    y_max_auto = max(
        float(actual_total.max()),
        float(res.fore_total_hi.max()) if res.fore_total_hi is not None else 0.0,
    )
    ax.set_ylim(
        y_min if y_min is not None else y_min_auto,
        y_max if y_max is not None else y_max_auto,
    )
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=300)

    if diagnostics:
        ax2 = ax.twinx()
        ax2.plot(
            res.years_fore,
            diagnostics.uncertainty_width,
            "k--",
            alpha=0.3,
            label="Uncertainty Width",
        )
        ax2.set_ylabel("Relative Uncertainty Width")

        plt.title(
            f"Total Incidents Forecast with Uncertainty\n" + f"(90% PI shown in light red)"
        )
        plt.tight_layout()
