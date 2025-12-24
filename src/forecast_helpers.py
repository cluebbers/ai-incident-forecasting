# forecast_helpers.py
# Reusable helpers to forecast incident counts by category with YTD assimilation, surge-aware shares,
# monotone totals, and coherent category allocations.
#
# Usage (example):
#   from forecast_helpers import ForecastConfig, forecast_by_category, plot_total_panel, plot_category_panels
#   cfg = ForecastConfig(YTD_YEAR=2025)
#   result = forecast_by_category(merged_df, date_col="date", cat_col="Risk Domain", config=cfg)
#   plot_total_panel(result)
#   plot_category_panels(result)
#
# You can reuse across columns:
#   for c in ["Risk Domain", "Actor", "Failure Mode"]:
#       res[c] = forecast_by_category(merged_df, "date", c, cfg)
#
# Notes:
# - Works on row-counts by default. To forecast on weighted totals, pass weight_col="my_value".
# - Avoids hard dependencies on your outer notebook variables.
# - Keeps defaults aligned to your current experiment; tune in ForecastConfig.

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

# ----------------------
# Configuration
# ----------------------


@dataclass
class ForecastConfig:
    # Simulation + randomness
    B_SIM: int = 2000  # Increased from 1000 for more stable estimates
    RANDOM_SEED: int = 42

    # Share blending / dominance control
    TAU_YEARS: float = 5.0
    W_HIST_MAX: float = 0.5
    W_UNIF_MAX: float = 0.2

    # Surge handling
    RECENT_WIN: int = 4
    CAT_BOOST_SURGE: float = 3.0
    SHRINK_SURGE: float = 0.3

    # Growth allocation constraints
    GROWTH_FLOOR_FRAC: float = 0.05
    MIN_ABS_GROWTH: float = 1e-6
    MOM_TILT_MAX: float = 0.3
    MOM_TAU: float = 5.0

    # Spline / GLM
    MAX_KNOTS_TOTAL: int = 8
    MAX_KNOTS_SHARES: int = 10
    ALPHA_TOTAL: float = 1.0

    # YTD
    YTD_YEAR: int = 2025
    YTD_METHOD: str = "ratio_to_average"
    YTD_MIN_SHARE: float = 0.35

    # Continuity floors (first forecast year)
    FIRST_YEAR_TOTAL_MIN_INC: float = 1.0
    FIRST_YEAR_CAT_MIN_INC: float = 1.0

    # Share + growth noise for simulations
    SHARE_KAPPA: float = 30.0
    GROWTH_NB_PHI: float = 0.3

    # Forecast horizon end (inclusive upper bound for index creation)
    END_YEAR: int = 2031

    # Backtesting
    MONOTONE_TOTALS: bool = True  # use isotonic after Poisson GAM
    NO_DIP_FIRST_YEAR: bool = True  # enforce y_{N} >= y_{N-1}
    NONDECREASING_CATEGORIES: bool = True
    ASSIMILATE_YTD: bool = False  # always False in backtest
    RECENT_WIN: int = 4

    # Uncertainty quantification parameters
    CONFIDENCE_LEVEL: float = 0.90  # Default 90% prediction intervals
    BOOTSTRAP_ITERS: int = 500  # For parameter uncertainty
    MIN_PI_WIDTH: float = 0.2  # Minimum relative width of prediction intervals
    GROWTH_UNCERTAINTY_FACTOR: float = 1.2  # Inflation factor for growth variance

    # Residual diagnostics
    STORE_RESIDUALS: bool = True  # Keep residuals for diagnostics
    MAX_ZSCORE: float = 4.0  # Cap standardized residuals


@dataclass
class ForecastResult:
    # Core outputs
    fore_df: pd.DataFrame
    fore_lo: pd.DataFrame
    fore_hi: pd.DataFrame
    fore_total: pd.Series
    fore_total_lo: pd.Series
    fore_total_hi: pd.Series

    # Context
    actual_by_year_full: pd.DataFrame
    actual_by_ym: pd.DataFrame
    cats: List[str]
    years_fore: np.ndarray
    last_year: int
    have_ytd: bool
    last_month: Optional[int]
    ytd_actual_total: Optional[float]
    ytd_year: int
    cat_col: str
    date_col: str
    total_sims: Optional[np.ndarray] = None


@dataclass
class ForecastDiagnostics:
    """Track forecast quality metrics and diagnostics"""

    residuals: np.ndarray
    z_scores: np.ndarray
    coverage_stats: Dict[str, float]
    parameter_uncertainty: Dict[str, np.ndarray]
    convergence_metrics: Dict[str, float]

    def plot_diagnostics(self):
        """Plot diagnostic visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # QQ plot of standardized residuals
        stats.probplot(self.z_scores, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title("Normal Q-Q Plot of Standardized Residuals")

        # Residuals vs fitted
        axes[0, 1].scatter(self.predicted_values, self.residuals)
        axes[0, 1].axhline(y=0, color="r", linestyle="-", alpha=0.3)
        axes[0, 1].set_xlabel("Fitted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Fitted")

        # Coverage plot
        nominal = np.array([0.5, 0.8, 0.9, 0.95])
        empirical = np.array([self.coverage_stats[f"{int(100*x)}%"] for x in nominal])
        axes[1, 0].plot(nominal, empirical, "bo-")
        axes[1, 0].plot([0, 1], [0, 1], "r--", alpha=0.3)
        axes[1, 0].set_xlabel("Nominal Coverage")
        axes[1, 0].set_ylabel("Empirical Coverage")
        axes[1, 0].set_title("Interval Coverage Calibration")

        plt.tight_layout()
        return fig


# ----------------------
# Low-level helpers
# ----------------------


def _make_features(
    n_knots: int, degree: int = 3, extrapolation: str = "linear", **kwargs
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots, degree=degree, extrapolation=extrapolation
                ),
                [0],
            ),
            ("linear", "passthrough", [0]),
        ]
    )


def _scale_years(year_array, center):
    return ((np.asarray(year_array) - center) / 10.0).reshape(-1, 1)


def allocate_growth(
    total_vec: np.ndarray,
    raw_mat: np.ndarray,
    floor_frac: float = 0.05,
    min_abs: float = 1e-6,
    mom_weights: Optional[np.ndarray] = None,
    mom_tilt_max: float = 0.3,
    mom_tau: float = 5.0,
    years: Optional[np.ndarray] = None,
    last_train: Optional[int] = None,
) -> np.ndarray:
    """
    Turn raw category guidance (per-year shares * totals) into coherent category counts
    with floors and momentum-tilting. Preserves monotonic non-decreasing path per category.
    """
    T, K = raw_mat.shape
    out = np.zeros_like(raw_mat, dtype=float)
    out[0] = np.maximum(raw_mat[0], 1e-9)
    out[0] *= total_vec[0] / max(out[0].sum(), 1e-12)
    tiny = 1e-12

    for t in range(1, T):
        prev = out[t - 1]
        T_prev, T_now = prev.sum(), total_vec[t]
        G = max(T_now - T_prev, 0.0)

        if G > 0:
            per_cat_floor = max(min_abs, floor_frac * G / K)
            total_floor = per_cat_floor * K
            if total_floor > G:
                per_cat_floor *= G / total_floor
        else:
            per_cat_floor = min_abs

        baseline = prev + per_cat_floor
        S_base = baseline.sum()
        if S_base > T_now:
            shrink = (S_base - T_now) / K
            per_cat_floor = max(0.0, per_cat_floor - shrink)
            baseline = prev + per_cat_floor
            S_base = baseline.sum()

        remaining = T_now - S_base
        shares_t = raw_mat[t] / max(raw_mat[t].sum(), 1e-12)

        if mom_weights is not None and years is not None and last_train is not None:
            h = max(0.0, years[t] - last_train)
            tilt = mom_tilt_max * (1.0 - np.exp(-h / mom_tau))
            shares_t = (1.0 - tilt) * shares_t + tilt * mom_weights

        alloc = remaining * shares_t if remaining > 0 else np.zeros_like(prev)
        c = baseline + alloc

        resid = c.sum() - T_now
        if abs(resid) > 1e-10:
            j = np.argmax(c - prev)
            c[j] -= resid
            c[j] = max(c[j], prev[j] + tiny)

        c = np.maximum(c, prev + tiny)
        out[t] = c

    return out


def _avg_cum_share_up_to(m: int, series: pd.Series) -> float:
    if series.empty:
        return np.nan
    s = series.copy()
    # group by (year, month) table
    s = s.groupby([s.index.year, s.index.month]).sum().unstack(1).fillna(0.0)
    cum = s.cumsum(axis=1)
    tot = s.sum(axis=1).replace(0, np.nan)
    shares = (cum.div(tot, axis=0)).mean(axis=0)
    return float(shares.get(m, np.nan))


# ----------------------
# Core pipeline
# ----------------------


def _build_actuals(
    df: pd.DataFrame,
    date_col: str,
    cat_col: str,
    weight_col: Optional[str],
    ytd_year: int,
) -> Dict[str, Any]:
    if date_col not in df.columns:
        raise KeyError(f"'{date_col}' not found in DataFrame.")
    if cat_col not in df.columns:
        raise KeyError(f"'{cat_col}' not found in DataFrame.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    d["__year"] = d[date_col].dt.year
    d["__month"] = d[date_col].dt.month
    d["__cat"] = d[cat_col].astype("category")

    # value to aggregate: 1 per row by default or weight_col
    if weight_col and weight_col in d.columns:
        d["__val"] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0.0)
    else:
        d["__val"] = 1.0

    cats_all = d["__cat"].cat.categories

    # Annual totals wide
    annual = (
        d.groupby(["__year", "__cat"])["__val"]
        .sum()
        .unstack("__cat")
        .reindex(columns=cats_all)
        .fillna(0.0)
    )

    # Monthly totals wide (first of month index)
    monthly = (
        d.groupby([pd.Grouper(key=date_col, freq="MS"), "__cat"])["__val"]
        .sum()
        .unstack("__cat")
        .reindex(columns=cats_all)
        .fillna(0.0)
    )
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month

    # YTD bookkeeping
    ymask = monthly["year"] == ytd_year
    months_obs = sorted(monthly.loc[ymask, "month"].unique().tolist())
    have_ytd = len(months_obs) > 0
    last_month = max(months_obs) if have_ytd else None

    return dict(
        annual=annual,
        monthly=monthly,
        cats_all=cats_all,
        have_ytd=have_ytd,
        last_month=last_month,
    )


def _annualize_ytd(
    annual: pd.DataFrame, monthly: pd.DataFrame, cats_all: pd.Index, cfg: ForecastConfig
) -> Dict[str, Any]:

    # Skip YTD assimilation if flag is off
    if not cfg.ASSIMILATE_YTD:
        return dict(
            annual=annual,
            monthly=monthly,
            have_ytd=False,
            last_month=None,
            ytd_actual_total=None,
        )

    # Use history strictly before YTD_YEAR
    hist_monthly = monthly[monthly["year"] < cfg.YTD_YEAR].copy()
    monthly_total = hist_monthly[cats_all].sum(axis=1)
    hist_monthly["total"] = monthly_total

    have_ytd = (monthly["year"] == cfg.YTD_YEAR).any()
    months_obs = sorted(
        monthly.loc[monthly["year"] == cfg.YTD_YEAR, "month"].unique().tolist()
    )
    last_month = max(months_obs) if (len(months_obs) > 0) else None

    ytd_actual_total = None
    est_cats_ytd = None

    if have_ytd and last_month is not None:
        total_series_hist = hist_monthly.set_index(hist_monthly.index)["total"]
        cum_share_total = _avg_cum_share_up_to(last_month, total_series_hist)
        if not np.isfinite(cum_share_total) or cum_share_total < cfg.YTD_MIN_SHARE:
            cum_share_total = cfg.YTD_MIN_SHARE

        # YTD total (sum over months observed in YTD year)
        ytd_total = (
            monthly.loc[
                (monthly["year"] == cfg.YTD_YEAR) & (monthly["month"] <= last_month),
                cats_all,
            ]
            .sum(axis=1)
            .sum()
        )
        ytd_actual_total = float(ytd_total)
        est_total_full = ytd_total / max(cum_share_total, 1e-9)

        # Per-category YTD sums
        ytd_cats = monthly.loc[
            (monthly["year"] == cfg.YTD_YEAR) & (monthly["month"] <= last_month),
            cats_all,
        ].sum()

        # Baseline shares from previous full year, else uniform
        prev_year = cfg.YTD_YEAR - 1
        if prev_year in annual.index:
            base_sh = annual.loc[prev_year]
            base_sh = (
                (base_sh / base_sh.sum()).values
                if base_sh.sum() > 0
                else np.full(len(cats_all), 1.0 / len(cats_all))
            )
        else:
            base_sh = np.full(len(cats_all), 1.0 / len(cats_all))

        remainder = max(est_total_full - ytd_cats.sum(), 0.0)
        est_cats_ytd = ytd_cats.values + remainder * base_sh
        est_cats_ytd = np.maximum(est_cats_ytd, 0.0)

        # Inject into annual table for continuity
        annual.loc[cfg.YTD_YEAR, cats_all] = est_cats_ytd

    return dict(
        annual=annual,
        monthly=monthly,
        have_ytd=have_ytd,
        last_month=last_month,
        ytd_actual_total=ytd_actual_total,
    )


def _add_hinge(X_scaled, pivot_year, t_mean):
    """Add hinge feature at pivot year for trend change"""
    # Convert scaled years back to original scale
    yr = X_scaled[:, 0] * 10.0 + t_mean
    # Create hinge feature
    hinge = np.maximum(0.0, yr - pivot_year)
    # Return both original scaled year and hinge feature
    return np.column_stack([X_scaled[:, 0], hinge])


def _fit_totals_and_shares(
    annual: pd.DataFrame, monthly: pd.DataFrame, cats_all: pd.Index, cfg: ForecastConfig
) -> Dict[str, Any]:
    # ---------- Totals ----------
    annual_total = annual.sum(axis=1).rename("total").to_frame()

    # Normalize years to int array
    if isinstance(annual_total.index, pd.DatetimeIndex):
        years_hist = np.sort(annual_total.index.year.values.astype(int))
    else:
        years_hist = np.sort(annual_total.index.values.astype(int))

    if years_hist.size < 2:
        raise ValueError("Need at least two years of data for totals.")

    t_mean = years_hist.mean()

    # For total model: use both year and hinge features
    X_hist_total = _scale_years(years_hist, t_mean)
    X_hist_total = _add_hinge(X_hist_total, pivot_year=2021, t_mean=t_mean)

    # Align target robustly
    y_total = annual_total.reindex(years_hist)["total"].astype(float).to_numpy()

    # ---------- Shares ----------
    # Long table from 'annual' (year x category counts)
    long = annual.stack().rename("count").reset_index()
    # Ensure expected column labels exist: year, cat
    if "year" not in long.columns:
        long = long.rename(columns={long.columns[0]: "year"})
    if "cat" not in long.columns:
        long = long.rename(columns={long.columns[1]: "cat"})

    # Keep only training years
    long = long[
        long["year"].between(int(years_hist.min()), int(years_hist.max()))
    ].copy()

    # Recency weights for totals (emphasize last RECENT_WIN years)
    win_tot = min(getattr(cfg, "RECENT_WIN", 4), max(1, len(years_hist)))
    last_y = int(years_hist.max())
    w_year = np.where(
        years_hist >= (last_y - win_tot + 1),
        np.exp((years_hist - (last_y - win_tot + 1)) / 1),
        1.0,
    ).astype(float)

    # total_model = Pipeline([
    #     ("features", _make_features(n_knots_total, degree=3, extrapolation="linear")),
    #     ("glm", PoissonRegressor(alpha=cfg.ALPHA_TOTAL, max_iter=5000))
    # ])
    total_model = Pipeline(
        [
            ("id", FunctionTransformer(lambda X: X, validate=False)),
            (
                "glm",
                PoissonRegressor(alpha=max(0.05, cfg.ALPHA_TOTAL * 0.1), max_iter=5000),
            ),
        ]
    )

    total_model.fit(X_hist_total, y_total, glm__sample_weight=w_year)

    # ---------- Shares ----------
    n_long = len(long)
    recency_w = np.ones(n_long, dtype=float)  # always defined

    last_year = int(years_hist.max())

    if n_long > 0:
        win = min(getattr(cfg, "RECENT_WIN", 4), max(1, np.unique(long["year"]).size))
        recency_w = np.where(
            long["year"] >= (last_year - win + 1),
            np.exp((long["year"] - (last_year - win + 1)) / 1.5),
            1.0,
        ).astype(float)

        # Optional extra weight for YTD year if present
        if hasattr(cfg, "YTD_YEAR") and (long["year"] == cfg.YTD_YEAR).any():
            recency_w = np.where(
                long["year"] == cfg.YTD_YEAR, recency_w * 1.5, recency_w
            )

    # Surge mask from recent share slopes (computed on annual table)
    hist_table = annual.reindex(columns=cats_all)
    hist_tot = hist_table.sum(axis=1).replace(0, np.nan)
    hist_shares_all = (hist_table.T / hist_tot).T.fillna(0.0)

    if hist_shares_all.shape[0] >= 2:
        win2 = min(getattr(cfg, "RECENT_WIN", 4), hist_shares_all.shape[0] - 1)
        share_slope = (
            hist_shares_all.iloc[-1] - hist_shares_all.iloc[-win2 - 1]
        ) / max(1, win2)
    else:
        share_slope = pd.Series(0.0, index=hist_shares_all.columns)

    slope_thresh = np.quantile(share_slope, 0.75) if len(share_slope) > 0 else 0.0
    surge_mask = (share_slope > slope_thresh) & (hist_shares_all.iloc[-1] > 0.05)

    def boost_for(cat):
        return cfg.CAT_BOOST_SURGE if surge_mask.get(cat, False) else 1.0

    if n_long == 0:
        raise ValueError("Share model training data is empty; check training window.")

    cat_w = long["cat"].map(lambda c: boost_for(c)).astype(float).values
    # w_cls_adj = long["count"].astype(float).values * recency_w * cat_w
    size_w = np.sqrt(
        np.maximum(1.0, long["count"].astype(float).values)
    )  # or np.log1p(...)
    w_raw = size_w * recency_w * cat_w
    cap = np.percentile(w_raw, 95)  # tame the 395+ tail
    w_cls_adj = np.minimum(w_raw, cap)
    # normalize to mean 1 to keep solver well-conditioned
    w_cls_adj = w_cls_adj * (w_cls_adj.size / w_cls_adj.sum())

    # Guard: sample_weight length must match samples
    assert w_cls_adj.shape[0] == n_long, "sample_weight length must match X,y"

    n_knots_shares = max(4, min(cfg.MAX_KNOTS_SHARES, np.unique(long["year"]).size))
    X_share = _scale_years(long["year"].values, t_mean)
    y_cls = long["cat"].values

    share_model = make_pipeline(
        SplineTransformer(n_knots=n_knots_shares, degree=3, extrapolation="linear"),
        LogisticRegression(
            multi_class="multinomial", C=1.0, max_iter=5000, solver="lbfgs"
        ),
    )
    share_model.fit(X_share, y_cls, logisticregression__sample_weight=w_cls_adj)

    cats = share_model.named_steps["logisticregression"].classes_
    K = len(cats)

    # Historical average shares for blending
    avg_hist_share = hist_shares_all.reindex(columns=cats).mean(axis=0).values
    avg_hist_share = np.maximum(avg_hist_share, 1e-12)
    avg_hist_share /= avg_hist_share.sum()

    # Momentum weights from positive recent share slopes
    mom_raw = np.maximum(share_slope.reindex(cats).fillna(0.0).values, 0.0)
    mom_w = np.full(K, 1.0 / K) if mom_raw.sum() == 0 else mom_raw / mom_raw.sum()
    # Print diagnostics before return
    print("years_hist:", years_hist[-6:])
    print("y_total:", y_total[-6:])
    print("w_year:", np.round(w_year[-6:], 3))
    print(
        "share weight quantiles:",
        np.quantile(w_cls_adj, [0, 0.25, 0.5, 0.75, 0.9, 0.99]),
    )

    # Return complete dictionary with required keys
    results = {
        "total_model": total_model,
        "share_model": share_model,
        "cats": list(cats),
        "avg_hist_share": avg_hist_share,
        "mom_w": mom_w,
        "last_year": last_year,
        "t_mean": t_mean,
        "y_total": y_total,
    }
    return results


def _bootstrap_uncertainty(model, X, y, n_boot=500):
    """Estimate parameter uncertainty via bootstrapping"""
    n = len(y)
    boot_coefs = []

    for _ in range(n_boot):
        # Residual bootstrap with non-negative constraint
        fitted = model.predict(X)
        resid = y - fitted
        boot_resid = np.random.choice(resid, size=n, replace=True)
        boot_y = fitted + boot_resid

        # Ensure non-negative values for Poisson regression
        boot_y = np.maximum(boot_y, 0.1)  # Small positive lower bound

        # Refit model
        boot_model = clone(model)
        try:
            boot_model.fit(X, boot_y)

            # Extract coefficients
            if hasattr(boot_model, "coef_"):
                boot_coefs.append(boot_model.coef_.copy())
        except (ValueError, np.linalg.LinAlgError):
            # Skip this bootstrap sample if fitting fails
            continue

    return np.array(boot_coefs) if boot_coefs else np.array([]).reshape(0, X.shape[1])


def _make_forecasts(
    models: Dict[str, Any],
    annual: pd.DataFrame,
    monthly: pd.DataFrame,
    cats_all: pd.Index,
    cfg: ForecastConfig,
) -> Tuple[ForecastResult, ForecastDiagnostics]:

    # Extract models
    total_model = models["total_model"]
    share_model = models["share_model"]

    # Pull auxiliary items returned from the fitter (with safe defaults)
    cats = models.get("cats", list(cats_all))
    avg_hist_share = np.asarray(
        models.get("avg_hist_share", np.full(len(cats), 1.0 / max(1, len(cats))))
    )
    mom_w = models.get("mom_w", None)
    last_year = int(
        models.get(
            "last_year",
            int(annual.index.max()) if len(annual.index) > 0 else cfg.YTD_YEAR,
        )
    )
    y_total = models.get("y_total", annual.sum(axis=1).astype(float).to_numpy())

    # Forecast horizon
    year_start = int(annual.index.min())
    year_end = cfg.END_YEAR
    years_fore = np.arange(year_start, year_end)

    # Define T (number of forecast periods)
    T = len(years_fore)

    # Get feature scaling center
    t_mean = models.get("t_mean", years_fore.mean())
    # Get training years for bootstrap
    if isinstance(annual.index, pd.DatetimeIndex):
        years_hist = np.sort(annual.index.year.values.astype(int))
    else:
        years_hist = np.sort(annual.index.values.astype(int))

    # Create training features for bootstrap (same as used in fitting)
    X_hist_total = _scale_years(years_hist, t_mean)
    X_hist_total = _add_hinge(X_hist_total, pivot_year=2021, t_mean=t_mean)

    # Create features for forecasting
    X_fore = _scale_years(years_fore, t_mean)
    X_fore = _add_hinge(X_fore, pivot_year=2021, t_mean=t_mean)

    # Create features for forecasting - separate for total and share models
    X_fore_total = _add_hinge(
        _scale_years(years_fore, t_mean), pivot_year=2021, t_mean=t_mean
    )
    X_fore_share = _scale_years(years_fore, t_mean)  # Single feature for share model
    # Generate predictions
    total_raw = total_model.predict(X_fore_total)
    proba = share_model.predict_proba(X_fore_share)

    # Apply monotonicity constraints if requested
    if cfg.MONOTONE_TOTALS:
        iso_total = IsotonicRegression(increasing=True, y_min=1e-8)
        total_fore = iso_total.fit_transform(years_fore, total_raw)
        for t in range(1, T):
            if total_fore[t] <= total_fore[t - 1]:
                total_fore[t] = total_fore[t - 1] + 1e-9
    else:
        total_fore = total_raw

    # Hard continuity for first forecast year (optional)
    y0 = int(last_year) + 1
    annual_total = annual.sum(axis=1)
    if cfg.NO_DIP_FIRST_YEAR and (last_year in annual.index) and (y0 in years_fore):
        last_actual_total = float(annual_total.loc[last_year])
        idx0 = int(np.where(years_fore == y0)[0][0])
        if total_fore[idx0] < last_actual_total + cfg.FIRST_YEAR_TOTAL_MIN_INC:
            shift = (last_actual_total + cfg.FIRST_YEAR_TOTAL_MIN_INC) - total_fore[
                idx0
            ]
            total_fore[idx0:] = total_fore[idx0:] + shift

    # Shares: model + blending (history/uniform) with surge shrink
    proba = share_model.predict_proba(X_fore_share)
    proba = np.maximum(proba, 1e-12)
    proba = proba / proba.sum(axis=1, keepdims=True)

    def _boost_for(
        cat, surge_mask=None
    ):  # placeholder: already baked into fit via weights
        return 1.0

    shrink_vec = np.array(
        [cfg.SHRINK_SURGE if _boost_for(c) > 1.0 else 1.0 for c in cats]
    )
    horizons = np.maximum(0, years_fore - last_year).astype(float)
    w_fac = 1.0 - np.exp(-horizons / cfg.TAU_YEARS)
    uniform_share = np.full(len(cats), 1.0 / len(cats))

    proba_blend = np.empty_like(proba)
    for t in range(T):
        w_hist_t = cfg.W_HIST_MAX * w_fac[t] * shrink_vec
        w_unif_t = cfg.W_UNIF_MAX * w_fac[t] * shrink_vec
        w_model_t = 1.0 - (w_hist_t + w_unif_t)
        w_model_t = np.clip(w_model_t, 0.0, None)
        tw = w_model_t + w_hist_t + w_unif_t
        tw = np.where(tw <= 0, 1.0, tw)
        w_model_t /= tw
        w_hist_t /= tw
        w_unif_t /= tw
        p = w_model_t * proba[t] + w_hist_t * avg_hist_share + w_unif_t * uniform_share
        p = np.maximum(p, 1e-12)
        proba_blend[t] = p / p.sum()

    # Simulations for intervals
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    SHARE_KAPPA = cfg.SHARE_KAPPA
    GROWTH_NB_PHI = cfg.GROWTH_NB_PHI

    def draw_overdispersed_poisson(mean_vec, phi):
        if phi <= 0:
            return rng.poisson(lam=mean_vec)
        g = rng.gamma(shape=max(1e-6, 1.0 / phi), scale=phi, size=mean_vec.shape)
        lam_tilde = mean_vec * g
        return rng.poisson(lam=np.clip(lam_tilde, 1e-12, None))

    K = len(cats)
    cats_sim = np.empty((cfg.B_SIM, T, K), dtype=float)

    growth_mean = np.empty(T)
    growth_mean[0] = total_fore[0]
    growth_mean[1:] = np.diff(total_fore)

    # Bootstrap parameter uncertainty
    total_boot_coefs = _bootstrap_uncertainty(
        total_model,
        X_hist_total,
        y_total,  # Use training features, not forecast features
        n_boot=cfg.BOOTSTRAP_ITERS,
    )

    # Modified simulation loop
    for b in range(cfg.B_SIM):
        # Sample from parameter uncertainty
        boot_idx = rng.integers(0, cfg.BOOTSTRAP_ITERS)
        if boot_idx < len(total_boot_coefs):
            total_model.coef_ = total_boot_coefs[boot_idx]

        # Increased growth uncertainty
        growth_draws = draw_overdispersed_poisson(
            growth_mean * cfg.GROWTH_UNCERTAINTY_FACTOR, GROWTH_NB_PHI
        )
        total_sim = np.cumsum(growth_draws).astype(float)
        for t in range(1, T):
            if total_sim[t] <= total_sim[t - 1]:
                total_sim[t] = total_sim[t - 1] + 1e-9

        # Dirichlet shares around blended probabilities
        p_draw = np.empty((T, K), dtype=float)
        for t in range(T):
            alpha = np.maximum(1e-8, SHARE_KAPPA * proba_blend[t])
            p_draw[t] = rng.dirichlet(alpha)

        raw_guidance = total_sim[:, None] * p_draw
        cats_sim[b] = allocate_growth(
            total_sim,
            raw_guidance,
            floor_frac=cfg.GROWTH_FLOOR_FRAC,
            min_abs=cfg.MIN_ABS_GROWTH,
            mom_weights=None if horizons.max() == 0 else (mom_w),
            mom_tilt_max=cfg.MOM_TILT_MAX,
            mom_tau=cfg.MOM_TAU,
            years=years_fore,
            last_train=last_year,
        )

    q_lo, q_hi, q_med = 0.05, 0.95, 0.50
    lo_k = np.quantile(cats_sim, q_lo, axis=0)
    hi_k = np.quantile(cats_sim, q_hi, axis=0)
    med_k = np.quantile(cats_sim, q_med, axis=0)

    fore_df = pd.DataFrame(med_k, index=years_fore, columns=cats)
    fore_lo = pd.DataFrame(lo_k, index=years_fore, columns=cats)
    fore_hi = pd.DataFrame(hi_k, index=years_fore, columns=cats)

    # Coherent total from sims
    tot_sim = cats_sim.sum(axis=2)
    fore_total = pd.Series(
        np.quantile(tot_sim, q_med, axis=0), index=years_fore, name="Total"
    )
    fore_total_lo = pd.Series(np.quantile(tot_sim, q_lo, axis=0), index=years_fore)
    fore_total_hi = pd.Series(np.quantile(tot_sim, q_hi, axis=0), index=years_fore)

    # Strict continuity for categories (optional)
    tiny = 1e-12
    if (
        cfg.NONDECREASING_CATEGORIES
        and (last_year in annual.index)
        and (y0 in fore_df.index)
    ):
        prev_vec = annual.reindex(columns=cats).loc[last_year].values.astype(float)
        cat_min_inc = np.where(prev_vec >= 1.0, cfg.FIRST_YEAR_CAT_MIN_INC, tiny)
        base = prev_vec + cat_min_inc

        for c in cats:
            if c in fore_df.columns:
                fore_df.loc[y0, c] = max(fore_df.loc[y0, c], base[cats.get_loc(c)])
                fore_df[c] = np.maximum(
                    fore_df[c], fore_df[c].shift(1) + cat_min_inc[cats.get_loc(c)]
                )
            else:
                fore_df[c] = np.maximum(
                    fore_df[c], fore_df[c].shift(1) + cat_min_inc[cats.get_loc(c)]
                )

    # YTD labels for plotting
    have_ytd = (monthly["year"] == cfg.YTD_YEAR).any()
    last_month = None
    ytd_actual_total = None
    if have_ytd:
        mset = monthly.loc[monthly["year"] == cfg.YTD_YEAR, "month"].unique().tolist()
        if len(mset) > 0:
            last_month = int(max(mset))
            ytd_actual_total = (
                monthly.loc[
                    (monthly["year"] == cfg.YTD_YEAR)
                    & (monthly["month"] <= last_month),
                    cats_all,
                ]
                .sum(axis=1)
                .sum()
            )

    return ForecastResult(
        fore_df=fore_df,
        fore_lo=fore_lo,
        fore_hi=fore_hi,
        fore_total=fore_total,
        fore_total_lo=fore_total_lo,
        fore_total_hi=fore_total_hi,
        actual_by_year_full=annual.reindex(columns=cats),
        actual_by_ym=monthly.reindex(columns=list(cats) + ["year", "month"]),
        cats=list(cats),
        years_fore=years_fore,
        last_year=last_year,
        have_ytd=have_ytd,
        last_month=last_month,
        ytd_actual_total=(
            float(ytd_actual_total) if ytd_actual_total is not None else None
        ),
        ytd_year=cfg.YTD_YEAR,
        cat_col="",
        date_col="",
        total_sims=tot_sim,
    )


def forecast_by_category(
    df: pd.DataFrame,
    date_col: str,
    cat_col: str,
    config: ForecastConfig,
    weight_col: Optional[str] = None,
) -> ForecastResult:
    """
    One-call forecaster:
        - builds annual/monthly actuals
        - assimilates YTD (ratio-to-average with min cumulative share)
        - fits total & share models
        - simulates to produce median + 90% PI
        - enforces continuity
        - returns ForecastResult
    """
    built = _build_actuals(
        df,
        date_col=date_col,
        cat_col=cat_col,
        weight_col=weight_col,
        ytd_year=config.YTD_YEAR,
    )
    annual0 = built["annual"].copy()
    monthly0 = built["monthly"].copy()
    cats_all = built["cats_all"]

    ann = annual0.copy()
    mon = monthly0.copy()

    _ytd = _annualize_ytd(ann, mon, cats_all, config)
    annual = _ytd["annual"]
    monthly = _ytd["monthly"]

    models = _fit_totals_and_shares(annual, monthly, cats_all, config)
    res = _make_forecasts(models, annual, monthly, cats_all, config)
    # keep context labels
    res.cat_col = cat_col
    res.date_col = date_col
    return res


def apply_to_columns(
    df: pd.DataFrame,
    date_col: str,
    cat_cols: List[str],
    config: ForecastConfig,
    weight_col: Optional[str] = None,
) -> Dict[str, ForecastResult]:
    """
    Convenience: run the same forecasting pipeline over multiple categorical columns.
    Returns a dict {cat_col -> ForecastResult}
    """
    out: Dict[str, ForecastResult] = {}
    for c in cat_cols:
        out[c] = forecast_by_category(
            df, date_col=date_col, cat_col=c, config=config, weight_col=weight_col
        )
    return out


# ----------------------
# Plotting helpers
# ----------------------

### OG function
def plot_total_panel(
    res: ForecastResult, diagnostics: Optional[ForecastDiagnostics] = None
):
    """Enhanced total panel plot with clear uncertainty visualization"""
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6.25, 2.16))

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
        alpha=0.3,
        label=f"90% PI",
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
    ax.set_ylim(bottom=-50)
    ax.grid(True, lw=0.3, alpha=0.5)
    plt.tight_layout()
    output_path = os.path.join("../output/total_incidents.pdf")
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
    ax.set_xlim(x_min if x_min is not None else x_min_auto,
                x_max if x_max is not None else x_max_auto)
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
    output_path = os.path.join("../output/total_incidents.pdf")
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
        label=f"90% PI",
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
    output_path = os.path.join("../output/total_incidents.pdf")
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


def plot_category_panels(
    res: ForecastResult,
    top_k: int | None = None,
    max_year: Optional[int] = None,
    show_legend: bool = True,
):
    """
    One subplot per category (optionally only the top_k), shared y-lims,
    with actuals + forecast median + 90% PI. Creates one figure.

    Ranking for top_k:
      - Uses the latest available forecast median for each category.
      - Falls back to last available actual if forecast is missing.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cats_all = list(res.cats)

    # --- pick top_k by latest forecast median (fallback to last actual) ---
    if top_k is not None and top_k < len(cats_all):
        last_fore_year = (
            int(res.fore_df.index.max()) if len(res.fore_df.index) else None
        )
        scores = {}
        for c in cats_all:
            val = None
            if last_fore_year is not None and c in res.fore_df.columns:
                v = res.fore_df.loc[last_fore_year, c]
                val = float(v) if np.isfinite(v) else None
            if val is None:
                # fallback: last actual (or total actual sum if needed)
                if c in res.actual_by_year_full.columns:
                    # prefer last row actual
                    last_idx = res.actual_by_year_full.index.max()
                    v2 = res.actual_by_year_full.loc[last_idx, c]
                    val = (
                        float(v2)
                        if np.isfinite(v2)
                        else float(res.actual_by_year_full[c].sum())
                    )
                else:
                    val = 0.0
            scores[c] = val

        cats = [
            c
            for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
                :top_k
            ]
        ]
    else:
        cats = cats_all

    K = len(cats)

    # --- y-lims across selected categories ---
    ymins, ymaxs = [], []
    for c in cats:
        series_list = [
            res.fore_lo[c].values if c in res.fore_lo.columns else np.array([]),
            res.fore_hi[c].values if c in res.fore_hi.columns else np.array([]),
            res.fore_df[c].values if c in res.fore_df.columns else np.array([]),
            (
                res.actual_by_year_full[c].values
                if c in res.actual_by_year_full.columns
                else np.array([])
            ),
        ]
        vals = (
            np.concatenate(
                [arr[~np.isnan(arr)] for arr in series_list if np.size(arr) > 0]
            )
            if series_list
            else np.array([])
        )
        if vals.size > 0:
            ymins.append(vals.min())
            ymaxs.append(vals.max())
    ymin = max(0.0, (min(ymins) if ymins else 0.0))
    ymax = max(ymaxs) if ymaxs else (ymin + 1.0)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin_plot = max(-50, ymin - pad)
    ymax_plot = ymax + pad

    # --- plotting ---
    n_rows = K
    n_cols = 1
    plt.figure(constrained_layout=True)
    for j, c in enumerate(cats):
        ax = plt.subplot(n_rows, n_cols, j + 1)
        if c in res.fore_lo.columns and c in res.fore_hi.columns:
            ax.fill_between(
                res.years_fore,
                res.fore_lo[c].values,
                res.fore_hi[c].values,
                alpha=0.25,
                label="90% PI",
            )
        if c in res.fore_df.columns:
            ax.plot(
                res.fore_df.index,
                res.fore_df[c].values,
                "--",
                lw=1.8,
                label="forecast (median)",
            )
        if c in res.actual_by_year_full.columns:
            ax.scatter(
                res.actual_by_year_full.index,
                res.actual_by_year_full[c].values,
                s=12,
                alpha=0.95,
                label="actual/est",
            )
        if (
            res.have_ytd
            and res.last_month is not None
            and c in res.actual_by_ym.columns
        ):
            ytd_cat_val = res.actual_by_ym.loc[
                (res.actual_by_ym["year"] == res.ytd_year)
                & (res.actual_by_ym["month"] <= res.last_month),
                c,
            ].sum()
            ax.scatter([res.ytd_year], [ytd_cat_val], s=20, marker="x")
        subtitle = f"{str(c)}"
        ax.set_title(subtitle)
        ax.set_ylabel("Count")
        ax.set_xlabel("Year" if j == n_rows - 1 else "")
        right = max_year if max_year is not None else 2030
        ax.set_xlim(2006, right)
        ax.set_ylim(ymin_plot, ymax_plot)
        ax.grid(True, lw=0.3, alpha=0.5)
        if show_legend and j == 0:
            base_title = f"Top {K} of {len(cats_all)}" if (top_k is not None) else None
            if base_title:
                ax.legend(loc="upper left", fontsize=8, title=base_title)
            else:
                ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    output_path = os.path.join("../output/incidents_category.pdf")
    plt.savefig(output_path, dpi=300)


# ----------------------
# Summary table helper
# ----------------------


def make_summary_table(
    res: ForecastResult, actual_years: List[int] = [2021, 2022, 2023, 2024]
) -> pd.DataFrame:
    """
    Summary table with (by-category + total):
      - Actual 2021-2024
      - YTD + 2025 full-year estimate (if available)
      - Predictions from year after last_year onward
    Returns Int64-typed DataFrame rounded to integers.
    """
    last_year = res.last_year
    years_fore = res.years_fore
    cats = list(res.cats)

    # Columns
    cols: List[str] = [f"Actual_{y}" for y in actual_years]
    if res.have_ytd and res.last_month is not None:
        cols += [
            f"Actual_{res.ytd_year}_YTD_m\u2264{res.last_month}",
            f"Est_{res.ytd_year}_Full",
        ]
    else:
        cols += [f"Actual_{res.ytd_year}"]

    pred_start = last_year + 1
    pred_cols = [f"Pred_{y}" for y in years_fore if y >= pred_start]
    cols += pred_cols

    summary_table = pd.DataFrame(index=cats + ["Total"], columns=cols, dtype=float)

    # Fill historical actuals
    for y in actual_years:
        if y in res.actual_by_year_full.index:
            summary_table.loc[cats, f"Actual_{y}"] = (
                res.actual_by_year_full.reindex(columns=cats).loc[y].values
            )
            summary_table.loc["Total", f"Actual_{y}"] = res.actual_by_year_full.loc[
                y
            ].sum()

    # 2025 YTD + est full-year
    if res.have_ytd and res.last_month is not None:
        ytd_cats = res.actual_by_ym.loc[
            (res.actual_by_ym["year"] == res.ytd_year)
            & (res.actual_by_ym["month"] <= res.last_month),
            cats,
        ].sum()
        summary_table.loc[
            cats, f"Actual_{res.ytd_year}_YTD_m\u2264{res.last_month}"
        ] = ytd_cats.values
        summary_table.loc[
            "Total", f"Actual_{res.ytd_year}_YTD_m\u2264{res.last_month}"
        ] = ytd_cats.sum()

        # The annual table already contains the 2025 full-year estimate
        if res.ytd_year in res.actual_by_year_full.index:
            summary_table.loc[cats, f"Est_{res.ytd_year}_Full"] = (
                res.actual_by_year_full.reindex(columns=cats).loc[res.ytd_year].values
            )
            summary_table.loc["Total", f"Est_{res.ytd_year}_Full"] = (
                res.actual_by_year_full.loc[res.ytd_year].sum()
            )
    else:
        if res.ytd_year in res.actual_by_year_full.index:
            summary_table.loc[cats, f"Actual_{res.ytd_year}"] = (
                res.actual_by_year_full.reindex(columns=cats).loc[res.ytd_year].values
            )
            summary_table.loc["Total", f"Actual_{res.ytd_year}"] = (
                res.actual_by_year_full.loc[res.ytd_year].sum()
            )

    # Predictions
    for y in years_fore:
        if y >= (last_year + 1):
            col = f"Pred_{y}"
            if y in res.fore_df.index:
                summary_table.loc[cats, col] = res.fore_df.loc[y].values
            if y in res.fore_total.index:
                summary_table.loc["Total", col] = float(res.fore_total.loc[y])

    summary_table = summary_table.astype(float).round(0).astype("Int64")
    return summary_table

def setup_plot_style():
    """Set up consistent matplotlib styling for all notebooks"""
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        print("Warning: scienceplots not available, using default style")
    
    width = 6.25  # onecolumn-format
    width2 = 3.06  # twocolumn-format
    aspect_ratio = np.sqrt(2)  # DIN A
    height = width / aspect_ratio
    height2 = width2 / aspect_ratio 

    sns.set_palette("colorblind")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],  # ACM fonts: ["Linux Biolinum O", "Linux Biolinum", "Biolinum"]
        "figure.figsize": (width, height),
        "figure.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "figure.titlesize": 8,
        "legend.fontsize": 8,
        "ytick.labelsize": 8,
        "xtick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "axes.linewidth": 0.5,
    })
    
    return {
        'width': width,
        'width2': width2, 
        'height': height,
        'height2': height2,
        'aspect_ratio': aspect_ratio
    }
