"""
FX Factor Regression
====================
Regresses a strategy return series against the FX factor returns to decompose
beta exposures.  Produces a regression summary and four visualisation panels.

Regression model
----------------
    r_strategy(t) = alpha + sum_i [ beta_i * factor_i(t) ] + epsilon(t)

Standard errors are Newey-West HAC (heteroskedasticity and autocorrelation
consistent), with the lag order set to floor(4 * (T/100)^(2/9)) by default —
the standard rule of thumb for daily financial returns.

Outputs
-------
1. Regression summary printed to stdout (alpha, betas, t-stats, p-values, R²).
2. Four-panel figure saved to disk:
   a. Beta bar chart with 95% confidence intervals.
   b. Rolling 12-month betas for each factor.
   c. Cumulative actual vs. fitted returns.
   d. Cumulative residual (unexplained alpha) return.

Usage — as a module
-------------------
    from fx_factor_analysis.analysis.factor_regression import run_factor_regression
    results = run_factor_regression(strategy_returns, factor_returns)

Usage — CLI
-----------
    python -m fx_factor_analysis.analysis.factor_regression \\
        --strategy path/to/strategy.csv \\
        --factors  path/to/factor_returns.csv \\
        --output   path/to/output_dir

    The strategy CSV must have a date column (index) and one return column.
    The factors CSV is the output of main.py (date index, one column per factor).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

FACTOR_LABELS: dict[str, str] = {
    "mom_hilo":    "Mom: HiLo",
    "mom_ma":      "Mom: MA",
    "mom_price":   "Mom: Price",
    "mom_skew":    "Mom: Skew",
    "carry_yc":    "Carry: YC",
    "carry_fwd":   "Carry: Fwd",
    "carry_rate":  "Carry: Rate",
    "asset_bond":  "Asset: Bond",
    "asset_equity":"Asset: Equity",
    "value_neer":  "Value: NEER",
    "value_ppp":   "Value: PPP",
}

# Colour palette: one per factor category
CATEGORY_COLORS: dict[str, str] = {
    "mom":   "#4C72B0",  # blue
    "carry": "#DD8452",  # orange
    "asset": "#55A868",  # green
    "value": "#C44E52",  # red
}

def _factor_color(factor_name: str) -> str:
    prefix = factor_name.split("_")[0]
    return CATEGORY_COLORS.get(prefix, "#8C8C8C")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RegressionResults:
    """Full regression output, passed between analysis and plotting."""
    alpha: float
    alpha_se: float
    alpha_tstat: float
    alpha_pval: float

    betas: pd.Series           # index = factor names
    beta_se: pd.Series
    beta_tstat: pd.Series
    beta_pval: pd.Series
    beta_ci_low: pd.Series     # 95% CI lower
    beta_ci_high: pd.Series    # 95% CI upper

    r_squared: float
    adj_r_squared: float
    n_obs: int

    fitted: pd.Series          # in-sample fitted values (daily)
    residuals: pd.Series       # epsilon(t)
    strategy: pd.Series        # aligned strategy returns used in regression

    rolling_betas: pd.DataFrame  # shape (dates × factors), rolling-window estimates


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def _newey_west_lags(n: int) -> int:
    """Rule-of-thumb lag order: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _align(strategy: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join on dates, drop rows where either side has NaN."""
    combined = factors.join(strategy.rename("__strat__"), how="inner")
    combined = combined.dropna()
    y = combined["__strat__"]
    X = combined.drop(columns=["__strat__"])
    return y, X


def _ols_with_nw(y: pd.Series, X: pd.DataFrame, lags: Optional[int] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const)
    if lags is None:
        lags = _newey_west_lags(len(y))
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return result


def _rolling_betas(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 252,   # ~12 months of daily data
) -> pd.DataFrame:
    """
    Rolling OLS betas (no HAC, for speed).  Returns DataFrame shaped
    (dates × factors); NaN for the warm-up period.
    """
    records = {}
    dates = y.index

    for i in range(window, len(dates) + 1):
        y_w = y.iloc[i - window : i]
        X_w = X.iloc[i - window : i]
        X_c = sm.add_constant(X_w, has_constant="add")
        try:
            res = sm.OLS(y_w, X_c).fit()
            records[dates[i - 1]] = res.params.drop("const")
        except Exception:
            records[dates[i - 1]] = pd.Series(np.nan, index=X.columns)

    if not records:
        return pd.DataFrame(index=dates, columns=X.columns, dtype=float)

    roll = pd.DataFrame(records).T
    roll = roll.reindex(dates)
    return roll


def run_factor_regression(
    strategy: pd.Series,
    factors: pd.DataFrame,
    rolling_window: int = 252,
    nw_lags: Optional[int] = None,
) -> RegressionResults:
    """
    Run full-sample OLS + rolling OLS of strategy returns on factor returns.

    Parameters
    ----------
    strategy : pd.Series
        Daily strategy return series (DatetimeIndex, decimal returns).
    factors : pd.DataFrame
        Daily factor returns from build_all_factor_returns() (same convention).
    rolling_window : int
        Number of trading days for rolling beta window (default 252 ≈ 1Y).
    nw_lags : int, optional
        Override Newey-West lag count.  Default: auto rule-of-thumb.

    Returns
    -------
    RegressionResults
    """
    y, X = _align(strategy, factors)
    logger.info("Regression sample: %s → %s  (%d observations)", y.index[0].date(), y.index[-1].date(), len(y))

    result = _ols_with_nw(y, X, lags=nw_lags)

    params = result.params
    bse = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    ci = result.conf_int(alpha=0.05)  # 95% CI

    factor_names = X.columns.tolist()

    roll = _rolling_betas(y, X, window=rolling_window)

    return RegressionResults(
        alpha=params["const"],
        alpha_se=bse["const"],
        alpha_tstat=tvals["const"],
        alpha_pval=pvals["const"],

        betas=params[factor_names],
        beta_se=bse[factor_names],
        beta_tstat=tvals[factor_names],
        beta_pval=pvals[factor_names],
        beta_ci_low=ci.loc[factor_names, 0],
        beta_ci_high=ci.loc[factor_names, 1],

        r_squared=result.rsquared,
        adj_r_squared=result.rsquared_adj,
        n_obs=int(result.nobs),

        fitted=result.fittedvalues,
        residuals=result.resid,
        strategy=y,

        rolling_betas=roll,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(res: RegressionResults) -> None:
    """Print a clean regression summary table to stdout."""
    rows = [("alpha (annualised)", res.alpha * 252, res.alpha_se * np.sqrt(252),
             res.alpha_tstat, res.alpha_pval)]
    for f in res.betas.index:
        rows.append((
            FACTOR_LABELS.get(f, f),
            res.betas[f],
            res.beta_se[f],
            res.beta_tstat[f],
            res.beta_pval[f],
        ))

    header = f"{'Factor':<22} {'Beta':>10} {'Std Err':>10} {'t-stat':>10} {'p-value':>10}  {'Sig':>4}"
    sep = "-" * len(header)

    print()
    print("FX Factor Regression Results")
    print(sep)
    print(header)
    print(sep)
    for name, beta, se, t, p in rows:
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  {name:<20} {beta:>10.4f} {se:>10.4f} {t:>10.3f} {p:>10.4f}  {sig:>4}")
    print(sep)
    print(f"  {'R²':<20} {res.r_squared:>10.4f}")
    print(f"  {'Adj. R²':<20} {res.adj_r_squared:>10.4f}")
    print(f"  {'N observations':<20} {res.n_obs:>10d}")
    print(sep)
    print("  Standard errors: Newey-West HAC  |  *** p<0.01  ** p<0.05  * p<0.10")
    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _significance_markers(pvals: pd.Series) -> list[str]:
    return ["***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "")) for p in pvals]


def plot_results(
    res: RegressionResults,
    output_path: Optional[str] = None,
    title: str = "FX Factor Exposure",
) -> plt.Figure:
    """
    Four-panel visualisation:
      (A) Beta bar chart with 95% CI error bars
      (B) Rolling 12M betas over time
      (C) Cumulative actual vs. fitted returns
      (D) Cumulative residual (alpha) return

    Parameters
    ----------
    res : RegressionResults
    output_path : str, optional
        If provided, save figure to this path (PNG, 200 dpi).
    title : str
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_beta   = fig.add_subplot(gs[0, 0])
    ax_roll   = fig.add_subplot(gs[0, 1])
    ax_cumret = fig.add_subplot(gs[1, 0])
    ax_alpha  = fig.add_subplot(gs[1, 1])

    factors = res.betas.index.tolist()
    labels = [FACTOR_LABELS.get(f, f) for f in factors]
    colors = [_factor_color(f) for f in factors]

    # ------------------------------------------------------------------
    # Panel A: Beta bar chart
    # ------------------------------------------------------------------
    y_pos = np.arange(len(factors))
    betas = res.betas.values
    ci_low  = res.beta_ci_low.values
    ci_high = res.beta_ci_high.values
    xerr_neg = betas - ci_low
    xerr_pos = ci_high - betas
    sigs = _significance_markers(res.beta_pval)

    bars = ax_beta.barh(y_pos, betas, color=colors, alpha=0.80, edgecolor="white", height=0.6)
    ax_beta.errorbar(betas, y_pos, xerr=[xerr_neg, xerr_pos],
                     fmt="none", color="black", capsize=3, linewidth=1.2)
    ax_beta.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Significance markers to the right of CI
    for i, (beta, high, sig) in enumerate(zip(betas, ci_high, sigs)):
        if sig:
            x = max(abs(beta), abs(high)) * np.sign(beta) if beta != 0 else high
            ax_beta.text(ci_high[i] + 0.002, i, sig, va="center", fontsize=8, color="black")

    ax_beta.set_yticks(y_pos)
    ax_beta.set_yticklabels(labels, fontsize=9)
    ax_beta.set_xlabel("Beta coefficient", fontsize=9)
    ax_beta.set_title("(A)  Factor Betas  [95% CI, Newey-West SE]", fontsize=10, fontweight="bold")
    ax_beta.tick_params(axis="x", labelsize=8)

    # Colour legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat.capitalize())
                       for cat, c in CATEGORY_COLORS.items()]
    ax_beta.legend(handles=legend_elements, fontsize=7, loc="lower right",
                   framealpha=0.7, title="Category", title_fontsize=7)

    # ------------------------------------------------------------------
    # Panel B: Rolling betas
    # ------------------------------------------------------------------
    roll = res.rolling_betas.dropna(how="all")
    for factor in factors:
        if factor not in roll.columns:
            continue
        ax_roll.plot(roll.index, roll[factor],
                     label=FACTOR_LABELS.get(factor, factor),
                     color=_factor_color(factor),
                     linewidth=0.9, alpha=0.85)
    ax_roll.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax_roll.set_title("(B)  Rolling 12M Betas", fontsize=10, fontweight="bold")
    ax_roll.set_xlabel("")
    ax_roll.set_ylabel("Beta", fontsize=9)
    ax_roll.legend(fontsize=6.5, ncol=2, framealpha=0.7,
                   loc="upper left", title="Factor", title_fontsize=7)
    ax_roll.tick_params(axis="both", labelsize=8)
    ax_roll.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    # ------------------------------------------------------------------
    # Panel C: Cumulative actual vs. fitted
    # ------------------------------------------------------------------
    strat_cum = (1 + res.strategy).cumprod() - 1
    fitted_cum = (1 + res.fitted).cumprod() - 1

    ax_cumret.plot(strat_cum.index, strat_cum.values * 100,
                   label="Strategy", color="#2C3E50", linewidth=1.4)
    ax_cumret.plot(fitted_cum.index, fitted_cum.values * 100,
                   label="Factor model (fitted)", color="#E74C3C",
                   linewidth=1.1, linestyle="--")
    ax_cumret.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_cumret.set_title("(C)  Cumulative Return: Actual vs. Fitted", fontsize=10, fontweight="bold")
    ax_cumret.set_xlabel("")
    ax_cumret.set_ylabel("Cumulative return (%)", fontsize=9)
    ax_cumret.legend(fontsize=8, framealpha=0.7)
    ax_cumret.tick_params(axis="both", labelsize=8)
    ax_cumret.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax_cumret.axhline(0, color="black", linewidth=0.5)

    # ------------------------------------------------------------------
    # Panel D: Cumulative residual (alpha) return
    # ------------------------------------------------------------------
    alpha_cum = (1 + res.residuals).cumprod() - 1
    # Annualised alpha in title
    ann_alpha_pct = res.alpha * 252 * 100

    ax_alpha.plot(alpha_cum.index, alpha_cum.values * 100,
                  color="#27AE60", linewidth=1.2)
    ax_alpha.fill_between(alpha_cum.index, alpha_cum.values * 100,
                          alpha=0.15, color="#27AE60")
    ax_alpha.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_alpha.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax_alpha.set_title(
        f"(D)  Cumulative Residual (Alpha)  "
        f"[Ann. α = {ann_alpha_pct:+.2f}%"
        f"{'*' if res.alpha_pval < 0.10 else ''}]",
        fontsize=10, fontweight="bold"
    )
    ax_alpha.set_xlabel("")
    ax_alpha.set_ylabel("Cumulative residual (%)", fontsize=9)
    ax_alpha.tick_params(axis="both", labelsize=8)
    ax_alpha.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    # Footer with R² annotation
    r2_text = (f"R² = {res.r_squared:.3f}  |  Adj. R² = {res.adj_r_squared:.3f}  |  "
               f"N = {res.n_obs:,d} days")
    fig.text(0.5, 0.01, r2_text, ha="center", fontsize=9, color="#555555")

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info("Figure saved → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regress a strategy return series against FX factor returns."
    )
    parser.add_argument(
        "--strategy", required=True,
        help="Path to CSV with strategy returns (date index, one return column).",
    )
    parser.add_argument(
        "--factors", required=True,
        help="Path to factor_returns.csv produced by main.py.",
    )
    parser.add_argument(
        "--output", default="output",
        help="Directory to save figure and summary CSV.",
    )
    parser.add_argument(
        "--rolling-window", type=int, default=252,
        help="Trading days for rolling beta window (default 252 ≈ 1Y).",
    )
    parser.add_argument(
        "--title", default="FX Factor Exposure",
        help="Figure title.",
    )
    parser.add_argument(
        "--start", default=None,
        help="Trim analysis to on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Trim analysis to on/before this date (YYYY-MM-DD).",
    )
    return parser.parse_args()


def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    # If multiple columns, raise a helpful error
    raise ValueError(
        f"Strategy CSV has {df.shape[1]} columns; expected exactly 1. "
        f"Columns found: {df.columns.tolist()}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)

    strategy = _load_series(args.strategy)
    factors = pd.read_csv(args.factors, index_col=0, parse_dates=True)

    # Optional date trim
    if args.start:
        strategy = strategy[strategy.index >= args.start]
        factors  = factors[factors.index >= args.start]
    if args.end:
        strategy = strategy[strategy.index <= args.end]
        factors  = factors[factors.index <= args.end]

    results = run_factor_regression(
        strategy, factors, rolling_window=args.rolling_window
    )

    print_summary(results)

    fig_path = os.path.join(args.output, "factor_exposure.png")
    plot_results(results, output_path=fig_path, title=args.title)

    # Save betas to CSV
    beta_df = pd.DataFrame({
        "beta": results.betas,
        "std_err": results.beta_se,
        "t_stat": results.beta_tstat,
        "p_value": results.beta_pval,
        "ci_low_95": results.beta_ci_low,
        "ci_high_95": results.beta_ci_high,
    })
    beta_df.index = [FACTOR_LABELS.get(f, f) for f in beta_df.index]
    beta_csv = os.path.join(args.output, "factor_betas.csv")
    beta_df.to_csv(beta_csv)
    logger.info("Betas saved → %s", beta_csv)


if __name__ == "__main__":
    main()
