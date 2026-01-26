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
    y_min_hist: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_uncertainty: bool = True,
    output_name: str = "total_incidents.pdf",
    dpi: int = 300,
):
    """Two-panel view: historical fit (top) and full forecast (bottom)."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        constrained_layout=True,
        figsize=figsize if figsize is not None else (6.5, 6.5),
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    actual_total = res.actual_by_year_full.sum(axis=1)
    max_obs = float(actual_total.max())

    def _plot_panel(ax):
        if show_uncertainty:
            ax.fill_between(
                res.years_fore,
                res.fore_total_lo,
                res.fore_total_hi,
                alpha=0.3,
                label="90% PI",
            )
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
        ax.set_ylabel("Count")
        ax.grid(True, lw=0.3, alpha=0.5)

    _plot_panel(ax_top)
    _plot_panel(ax_bottom)

    top_x_min = x_min if x_min is not None else 2000
    top_x_max = 2025
    bottom_x_min = x_min if x_min is not None else 2000
    bottom_x_max = x_max if x_max is not None else 2030

    ticks = np.arange(bottom_x_min, bottom_x_max + 1, 5)
    ax_top.set_xticks(ticks)
    ax_bottom.set_xticks(ticks)

    ax_top.set_title("Historical fit")
    ax_top.set_xlim(top_x_min, top_x_max)
    ax_top.set_ylim(y_min_hist if y_min_hist is not None else 0, 1.2 * max_obs)

    ax_bottom.set_title("Full forecast")
    ax_bottom.set_xlim(bottom_x_min, bottom_x_max)
    if y_min is not None or y_max is not None:
        ax_bottom.set_ylim(y_min if y_min is not None else None, y_max if y_max is not None else None)
    ax_bottom.set_xlabel("Year")

    ax_top.legend()

    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=dpi)

    if diagnostics:
        ax2 = ax_bottom.twinx()
        ax2.plot(
            res.years_fore,
            diagnostics.uncertainty_width,
            "k--",
            alpha=0.3,
            label="Uncertainty Width",
        )
        ax2.set_ylabel("Relative Uncertainty Width")
