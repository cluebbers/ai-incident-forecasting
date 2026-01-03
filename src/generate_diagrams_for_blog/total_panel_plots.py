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


def _format_output_path(filename: str) -> str:
    safe_name = filename.strip() if filename else "total_incidents.pdf"
    if not safe_name:
        safe_name = "total_incidents.pdf"
    if "." not in os.path.basename(safe_name):
        safe_name = f"{safe_name}.pdf"
    return os.path.join("../output", safe_name)


def plot_total_panel_hinge(
    res: ForecastResult,
    diagnostics: Optional[ForecastDiagnostics] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_uncertainty: bool = True,
    hinge_year: Optional[int] = 2021,
    legend_loc: str = "upper right",
    y_offset_mult: float = 0.05,
    output_name: str = "total_incidents.pdf",
):
    """Mini chart with a hinge (kinked) trend line."""
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

    years_min = int(actual_total.index.min())
    years_max = int(actual_total.index.max())
    if hinge_year is None or hinge_year < years_min or hinge_year > years_max:
        hinge_year = int(np.median(actual_total.index))

    ax.scatter(
        actual_total.index,
        actual_total.values,
        s=18,
        label="Total actual",
    )

    if res.ytd_actual_total is not None and res.have_ytd and res.last_month is not None:
        ax.scatter(
            [res.ytd_year],
            [res.ytd_actual_total],
            s=28,
            marker="x",
            label=f"{res.ytd_year} YTD actual (m≤{res.last_month})",
        )

    years = actual_total.index.to_numpy(dtype=float)
    counts = actual_total.to_numpy(dtype=float)
    x = years - float(hinge_year)
    hinge = np.maximum(0.0, x)
    design = np.column_stack([np.ones_like(x), x, hinge])
    coef, _, _, _ = np.linalg.lstsq(design, counts, rcond=None)

    fit_years = np.arange(years_min, years_max + 1, dtype=float)
    fit_x = fit_years - float(hinge_year)
    fit_hinge = np.maximum(0.0, fit_x)
    fit_vals = coef[0] + coef[1] * fit_x + coef[2] * fit_hinge
    pre_mask = fit_years <= hinge_year
    post_mask = fit_years >= hinge_year
    ax.plot(
        fit_years[pre_mask],
        fit_vals[pre_mask],
        "-",
        lw=2,
        color="#1f77b4",
        label="Kinked trend",
    )
    ax.plot(
        fit_years[post_mask],
        fit_vals[post_mask],
        "--",
        lw=2,
        color="#1f77b4",
    )

    ax.axvline(hinge_year, color="#666666", lw=1, ls=":")
    ax.text(
        hinge_year,
        0.95,
        "hinge year",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
        color="#666666",
    )

    before_mid = (years_min + hinge_year) / 2.0
    after_mid = (hinge_year + years_max) / 2.0
    before_val = coef[0] + coef[1] * (before_mid - hinge_year)
    after_val = coef[0] + coef[1] * (after_mid - hinge_year) + coef[2] * max(
        0.0, after_mid - hinge_year
    )
    # ax.text(before_mid, before_val, "before", fontsize=8, color="#555555")
    # ax.text(after_mid, after_val, "after", fontsize=8, color="#555555")
    # ax.set_title(
    #     f"Total incidents ({res.cat_col}) with {res.ytd_year} YTD assimilation"
    # )
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(loc=legend_loc)
    x_min_auto = min(actual_total.index.min(), res.years_fore.min())
    x_max_auto = max(actual_total.index.max(), res.years_fore.max())
    ax.set_xlim(
        x_min if x_min is not None else x_min_auto,
        x_max if x_max is not None else x_max_auto,
    )
    y_min_auto = -50
    y_max_auto = max(
        float(actual_total.max()),
        float(res.fore_total_hi.max()) if res.fore_total_hi is not None else 0.0,
    )
    ax.set_ylim(
        y_min if y_min is not None else y_min_auto,
        y_max if y_max is not None else y_max_auto,
    )
    y_offset = y_offset_mult * (y_max_auto - y_min_auto)
    ax.text(
        after_mid,
        after_val - y_offset,
        "+ growth boost after 2021",
        fontsize=8,
        color="#555555",
    )
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=1000)

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


def plot_total_panel_possion_regression(
    res: ForecastResult,
    diagnostics: Optional[ForecastDiagnostics] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    show_uncertainty: bool = True,
    show_linear_fit: bool = False,
    figsize: Optional[tuple] = None,
    output_name: str = "total_incidents.pdf",
):
    """Mini chart: counts over time with Poisson GLM mean and uncertainty."""
    fig, ax = plt.subplots(
        constrained_layout=True, figsize=figsize if figsize is not None else (3.4, 2.0)
    )

    actual_total = res.actual_by_year_full.sum(axis=1)
    mean_series = res.fore_total

    if show_uncertainty:
        ax.fill_between(
            res.years_fore,
            res.fore_total_lo,
            res.fore_total_hi,
            color="#c6dbef",
            alpha=0.5,
            lw=0,
        )

    ax.plot(
        mean_series.index,
        mean_series.values,
        "--",
        lw=1.6,
        color="#1f77b4",
        label="Poisson Regression",
    )
    if show_linear_fit and len(actual_total) >= 2:
        x_vals = actual_total.index.to_numpy(dtype=float)
        y_vals = actual_total.to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        fit_years = np.array([x_vals.min(), x_vals.max()])
        fit_vals = slope * fit_years + intercept
        ax.plot(
            fit_years,
            fit_vals,
            "-",
            lw=1.2,
            color="#8c8c8c",
            label="Linear Regression",
        )
    ax.scatter(
        actual_total.index,
        actual_total.values,
        s=14,
        color="orange",
        zorder=3,
        label="Total Actual",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    x_min_auto = min(actual_total.index.min(), res.years_fore.min())
    x_max_auto = max(actual_total.index.max(), res.years_fore.max())
    ax.set_xlim(
        x_min if x_min is not None else x_min_auto,
        x_max if x_max is not None else x_max_auto,
    )
    y_min_auto = 0
    y_max_auto = max(
        float(actual_total.max()),
        float(res.fore_total_hi.max()) if res.fore_total_hi is not None else 0.0,
    )
    ax.set_ylim(
        y_min if y_min is not None else y_min_auto,
        y_max if y_max is not None else y_max_auto,
    )
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.tight_layout()
    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=1000)

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


def plot_total_panel_monte_carlo(
    res: ForecastResult,
    diagnostics: Optional[ForecastDiagnostics] = None,
    zoom_start_year: Optional[int] = None,
    max_year: Optional[int] = None,
    show_legend: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    sim_line_color: str = "#457A8418",
    show_sims: bool = False,
    max_sim_lines: int = 150,
    output_name: str = "total_incidents.pdf",
):
    """Enhanced total panel plot with clear uncertainty visualization"""
    fig_size = figsize if figsize is not None else (6.25, 2.16)
    fig, ax = plt.subplots(constrained_layout=True, figsize=fig_size)

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
    lo = res.fore_total_lo
    hi = res.fore_total_hi
    ax.fill_between(
        res.years_fore,
        lo,
        hi,
        color="#9ecae1",
        alpha=0.3,
        label="90% PI",
    )

    if show_sims and res.total_sims is not None and len(res.total_sims) > 0:
        total_sims = res.total_sims
        sim_count = min(max_sim_lines, total_sims.shape[0])
        if sim_count > 0:
            sim_idx = np.linspace(0, total_sims.shape[0] - 1, sim_count, dtype=int)
            for idx in sim_idx:
                ax.plot(
                    res.years_fore,
                    total_sims[idx],
                    color=sim_line_color,
                    lw=0.6,
                    zorder=2,
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
    if show_legend:
        ax.legend()
    if y_min is None and y_max is None:
        ax.set_ylim(bottom=-50)
    else:
        ax.set_ylim(bottom=y_min, top=y_max)
    if zoom_start_year is not None or max_year is not None:
        right = max_year if max_year is not None else max(res.years_fore)
        if zoom_start_year is not None:
            ax.set_xlim(left=zoom_start_year, right=right)
        else:
            ax.set_xlim(right=right)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=1000)

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


def plot_total_panel_logistic_regression(
    res: ForecastResult,
    diagnostics: Optional[ForecastDiagnostics] = None,
    x_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_uncertainty: bool = True,
    show_empirical_markers: bool = True,
    legend_loc: str = "upper left",
    legend_outside: bool = True,
    legend_bbox: tuple[float, float] = (1.02, 1.0),
    note_pos: tuple[float, float] = (0.98, 0.02),
    output_name: str = "total_incidents.pdf",
):
    """Stacked area chart of predicted category shares over time."""
    fig, ax = plt.subplots(
        constrained_layout=True, figsize=figsize if figsize is not None else (6.25, 2.16)
    )

    forecast_shares = res.fore_df.div(res.fore_df.sum(axis=1), axis=0)
    years = forecast_shares.index.to_numpy(dtype=float)
    categories = list(forecast_shares.columns)
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    ax.stackplot(
        years,
        [forecast_shares[c].to_numpy(dtype=float) for c in categories],
        labels=categories,
        colors=colors,
        alpha=0.85,
    )

    if show_empirical_markers:
        actual = res.actual_by_year_full.reindex(columns=categories).fillna(0)
        actual_shares = actual.div(actual.sum(axis=1).replace(0, np.nan), axis=0)
        for idx, cat in enumerate(categories):
            ax.plot(
                actual_shares.index,
                actual_shares[cat],
                linestyle="",
                marker="o",
                markersize=2,
                color=colors[idx],
                alpha=0.6,
                zorder=3,
            )

    ax.text(
        note_pos[0],
        note_pos[1],
        "shares sum to 1 (softmax)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666666",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Share")
    if legend_outside:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            frameon=False,
            fontsize=7,
            ncol=2,
        )
    else:
        ax.legend(loc=legend_loc, frameon=False, fontsize=7, ncol=2)

    x_min_auto = years.min()
    x_max_auto = years.max()
    ax.set_xlim(
        x_min if x_min is not None else x_min_auto,
        x_max if x_max is not None else x_max_auto,
    )
    ax.set_ylim(
        y_min if y_min is not None else 0.0,
        y_max if y_max is not None else 1.0,
    )
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.tight_layout()
    output_path = _format_output_path(output_name)
    plt.savefig(output_path, dpi=1000)
