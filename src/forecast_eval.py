import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forecast_helpers import ForecastConfig, forecast_by_category, ForecastResult

DEBUG = True  # Debug flag to control output


def debug(*args, **kwargs):
    """Helper function to print debug info"""
    if DEBUG:
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()


@dataclass
class BacktestResult:
    year: int
    actual: float
    predicted: float
    prediction_interval: Tuple[float, float]
    mae: float
    mape: float
    rmse: float


def backtest_forecast(
    df: pd.DataFrame,
    date_col: str,
    cat_col: str,
    test_years: list[int],
    base_config: ForecastConfig,
) -> list[BacktestResult]:
    """
    Performs backtesting of forecasting model on historical data.
    """
    results = []

    debug("\n" + "#" * 80)
    debug("BACKTEST EVALUATION STARTING")
    debug("#" * 80)

    # Print base config
    debug("\nBase Configuration:")
    for k, v in base_config.__dict__.items():
        if k.isupper():
            debug(f"- {k}: {v}")

    for test_year in test_years:
        debug(f"\nProcessing test year: {test_year}")
        debug("=" * 50)

        # Create training dataset up to test_year-1
        train_df = df[pd.to_datetime(df[date_col]).dt.year < test_year].copy()
        test_df = df[pd.to_datetime(df[date_col]).dt.year == test_year].copy()

        debug(f"\nData splits:")
        debug(f"Training data: {len(train_df)} rows")
        debug(f"Test data: {len(test_df)} rows")

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Configure forecast for this iteration
        config = ForecastConfig(
            YTD_YEAR=test_year,
            END_YEAR=test_year + 1,
            **{
                k: v
                for k, v in base_config.__dict__.items()
                if k not in ["YTD_YEAR", "END_YEAR"]
            },
        )

        debug("\nGenerating forecast...")
        res = forecast_by_category(train_df, date_col, cat_col, config)

        # Calculate metrics
        actual = float(
            test_df.groupby(pd.to_datetime(test_df[date_col]).dt.year)
            .size()
            .get(test_year, 0)
        )
        pred = float(res.fore_total[test_year].mean())

        # Get prediction interval
        pi_low = float(np.percentile(res.fore_total[test_year], 5))
        pi_high = float(np.percentile(res.fore_total[test_year], 95))

        # Calculate errors
        abs_err = abs(actual - pred)
        pct_err = abs_err / max(actual, 1.0) * 100
        rmse = np.sqrt(((actual - pred) ** 2))

        results.append(
            BacktestResult(
                year=test_year,
                actual=actual,
                predicted=pred,
                prediction_interval=(pi_low, pi_high),
                mae=abs_err,
                mape=pct_err,
                rmse=rmse,
            )
        )

    return results


def naive_baseline_forecast(
    df: pd.DataFrame, date_col: str, test_years: list[int]
) -> list[BacktestResult]:
    """
    Simple baseline model that predicts next year's count as:
    average(previous 3 years) * (1 + avg_growth_rate)
    """
    results = []

    for test_year in test_years:
        # Get previous years' data
        yearly_counts = (
            df[pd.to_datetime(df[date_col]).dt.year < test_year]
            .groupby(pd.to_datetime(df[date_col]).dt.year)
            .size()
        )

        if len(yearly_counts) < 2:
            continue

        # Calculate average growth rate
        growth_rates = yearly_counts.pct_change().dropna()
        avg_growth = growth_rates.mean()

        # Predict using last 3 years average * growth
        last_3y_avg = yearly_counts.tail(3).mean()
        pred = last_3y_avg * (1 + avg_growth)

        # Get actual
        actual = float(df[pd.to_datetime(df[date_col]).dt.year == test_year].shape[0])

        # Calculate errors
        abs_err = abs(actual - pred)
        pct_err = abs_err / max(actual, 1.0) * 100
        sq_err = (actual - pred) ** 2

        results.append(
            BacktestResult(
                year=test_year,
                actual=actual,
                predicted=pred,
                mae=abs_err,
                mape=pct_err,
                rmse=np.sqrt(sq_err),
                prediction_interval=(pred, pred),  # Baseline has no uncertainty
            )
        )

    return results


def _compute_coverage(
    res: ForecastResult, actuals: pd.Series, levels: List[float] = [0.5, 0.8, 0.9, 0.95]
) -> Dict[str, float]:
    """Compute empirical coverage at different confidence levels"""
    coverage = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = res.fore_total.quantile(alpha)
        hi = res.fore_total.quantile(1 - alpha)
        in_interval = (actuals >= lo) & (actuals <= hi)
        coverage[f"{int(100*level)}%"] = in_interval.mean()
    return coverage


def compute_empirical_coverage(results: List[BacktestResult]) -> float:
    """
    Compute empirical coverage as the fraction of actuals that fall within their prediction intervals.
    """
    if not results:
        return 0.0
    covered = [
        (r.prediction_interval[0] <= r.actual <= r.prediction_interval[1])
        for r in results
    ]
    return float(np.mean(covered))


def tune_uncertainty_params(
    df: pd.DataFrame,
    test_years: List[int],
    target_coverage: float = 0.90,
    max_rel_width: float = 0.5,
) -> ForecastConfig:
    """
    Tune uncertainty-related parameters by simple grid search to target desired coverage and sharpness.
    """
    # Build a base configuration
    try:
        base_year = max(test_years)
        base_cfg = ForecastConfig(YTD_YEAR=base_year, END_YEAR=base_year + 1)
    except Exception:
        # Fallback to default constructor if possible
        try:
            base_cfg = ForecastConfig()
        except Exception as e:
            raise RuntimeError("Unable to instantiate ForecastConfig for tuning") from e

    best_cfg = base_cfg
    best_score = float("inf")

    # Candidate configurations: vary common uncertainty parameters if present on config
    candidates: List[ForecastConfig] = [base_cfg]
    param_grid: Dict[str, List[float]] = {
        "NOISE_SCALE": [0.5, 0.75, 1.0, 1.25, 1.5],
        "UNCERTAINTY_SCALE": [0.5, 0.75, 1.0, 1.25, 1.5],
        "PI_SCALE": [0.5, 0.75, 1.0, 1.25, 1.5],
        "BOOTSTRAP_FRACTION": [0.5, 0.7, 0.9, 1.0],
    }

    for param, values in param_grid.items():
        if hasattr(base_cfg, param):
            for v in values:
                try:
                    candidates.append(replace(base_cfg, **{param: v}))
                except TypeError:
                    cfg = base_cfg
                    setattr(cfg, param, v)
                    candidates.append(cfg)

    # Evaluate candidates
    for cfg in candidates:
        results = backtest_forecast(df, "date", "Risk Domain", test_years, cfg)
        if not results:
            continue

        coverage = compute_empirical_coverage(results)
        sharpness = float(
            np.mean(
                [
                    (r.prediction_interval[1] - r.prediction_interval[0])
                    / max(r.predicted, 1.0)
                    for r in results
                ]
            )
        )

        coverage_error = abs(coverage - target_coverage)
        width_penalty = max(0.0, sharpness - max_rel_width)
        score = coverage_error + width_penalty

        if score < best_score:
            best_score = score
            best_cfg = cfg

    return best_cfg


def evaluate_sharpness(res: ForecastResult) -> Dict[str, float]:
    """Evaluate sharpness of prediction intervals"""
    total_pred = res.fore_total
    years = sorted(total_pred.keys())

    intervals = {}
    rel_widths = {}

    for year in years:
        pred = total_pred[year]
        q_lo = np.percentile(pred, 5)
        q_hi = np.percentile(pred, 95)
        width = q_hi - q_lo
        rel_width = width / np.mean(pred)

        intervals[year] = (q_lo, q_hi)
        rel_widths[year] = rel_width

    return {
        "intervals": intervals,
        "relative_widths": rel_widths,
        "mean_rel_width": np.mean(list(rel_widths.values())),
    }
