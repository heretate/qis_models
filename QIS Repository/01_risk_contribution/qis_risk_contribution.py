"""
qis_risk_contribution.py
------------------------
Analyses the theoretical risk contribution of adding a QIS strategy
(composite + 4 substrategies) to an existing portfolio at allocation
levels of 1–10% (levered: existing weights held fixed).

Metrics (all expressed as % contribution to total portfolio risk):
  - Annualised Volatility  : Euler decomposition  RC_i = w_i * (Σw)_i / (w'Σw)
  - VaR 95%               : Euler decomposition on sample covariance
  - CVaR 95%              : average loss beyond VaR threshold
  - Max Drawdown          : blended portfolio DD vs base portfolio DD (delta)

Outputs:
  - Printed summary tables per QIS series
  - Matplotlib charts: one per metric, lines per QIS series vs allocation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent   # CSVs live in repo root
CONFIDENCE   = 0.95
ALLOC_RANGE  = np.arange(0.01, 0.11, 0.01)          # 1% … 10%
TRADING_DAYS = 252

QIS_SERIES = [
    "qis_composite",
    "qis_momentum",
    "qis_value",
    "qis_carry",
    "qis_low_vol",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    holdings = pd.read_csv(DATA_DIR / "portfolio_holdings.csv")
    port_ret = pd.read_csv(DATA_DIR / "portfolio_returns.csv", parse_dates=["date"])
    qis_ret  = pd.read_csv(DATA_DIR / "qis_returns.csv",       parse_dates=["date"])

    port_ret = port_ret.set_index("date").sort_index()
    qis_ret  = qis_ret.set_index("date").sort_index()

    # align to overlapping dates
    overlap = port_ret.index.intersection(qis_ret.index)
    port_ret = port_ret.loc[overlap]
    qis_ret  = qis_ret.loc[overlap]

    tickers = holdings["ticker"].tolist()
    weights = holdings["weight"].values          # shape (N,)

    return tickers, weights, port_ret[tickers], qis_ret, overlap


# ── Risk helpers ──────────────────────────────────────────────────────────────
def portfolio_returns(ret_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Daily portfolio return series."""
    return ret_matrix @ weights


def annualised_vol(port_r: np.ndarray) -> float:
    return port_r.std() * np.sqrt(TRADING_DAYS)


def euler_vol_contribution(cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Euler decomposition of volatility.
    RC_i = w_i * (Σw)_i / sqrt(w'Σw)
    Returns array of % contributions summing to 1.
    """
    sigma_w   = cov @ weights
    port_var  = weights @ sigma_w
    port_vol  = np.sqrt(port_var)
    rc        = weights * sigma_w / port_vol        # absolute RC (daily)
    return rc / rc.sum()                            # % contribution


def euler_var_contribution(cov: np.ndarray, weights: np.ndarray, alpha: float = CONFIDENCE) -> np.ndarray:
    """
    Euler decomposition of Gaussian VaR.
    Under normality, VaR_alpha = z_alpha * sigma, so the decomposition
    is identical in % terms to the vol decomposition.
    """
    from scipy.stats import norm
    # % contributions are the same as vol under normality
    return euler_vol_contribution(cov, weights)


def cvar_contribution(ret_matrix: np.ndarray, weights: np.ndarray, alpha: float = CONFIDENCE) -> np.ndarray:
    """
    % CVaR contribution via component CVaR.
    Component CVaR_i = E[r_i | portfolio return <= VaR threshold].
    Scaled so contributions sum to 1.
    """
    port_r    = ret_matrix @ weights
    threshold = np.quantile(port_r, 1 - alpha)
    tail_mask = port_r <= threshold

    tail_contribs = (ret_matrix[tail_mask] * weights).mean(axis=0)  # E[w_i * r_i | tail]
    # negate so losses are positive; then normalise
    tail_contribs = -tail_contribs
    return tail_contribs / tail_contribs.sum()


def max_drawdown(r: np.ndarray) -> float:
    """Maximum drawdown of a return series."""
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return dd.min()      # most negative value


def qis_mdd_contribution(base_r: np.ndarray, blended_r: np.ndarray) -> float:
    """
    QIS % contribution to max drawdown = delta in portfolio MDD
    expressed as a fraction of the blended MDD.
    """
    mdd_base    = max_drawdown(base_r)
    mdd_blended = max_drawdown(blended_r)
    delta       = mdd_blended - mdd_base          # negative = worse DD
    if mdd_blended == 0:
        return 0.0
    return delta / abs(mdd_blended)               # % of blended MDD attributable to QIS


# ── Core analysis ─────────────────────────────────────────────────────────────
def analyse_qis_series(
    series_name: str,
    tickers: list,
    base_weights: np.ndarray,
    port_ret: pd.DataFrame,
    qis_series: pd.Series,
) -> pd.DataFrame:
    """
    For a single QIS return series, sweep allocations 1–10% and compute
    all four risk contributions.
    """
    N        = len(tickers)
    records  = []

    # base portfolio returns (no QIS)
    base_r   = portfolio_returns(port_ret.values, base_weights)

    for alloc in ALLOC_RANGE:
        # augmented weight vector: existing holdings unchanged, QIS added
        w_aug    = np.append(base_weights, alloc)
        ret_aug  = np.column_stack([port_ret.values, qis_series.values])

        # covariance of augmented return matrix
        cov_aug  = np.cov(ret_aug.T)                # (N+1) x (N+1)

        # blended portfolio returns
        blended_r = portfolio_returns(ret_aug, w_aug)

        # ── Vol contribution ──────────────────────────────────────────────────
        pct_vol  = euler_vol_contribution(cov_aug, w_aug)
        qis_vol  = pct_vol[-1] * 100

        # ── VaR contribution ──────────────────────────────────────────────────
        pct_var  = euler_var_contribution(cov_aug, w_aug)
        qis_var  = pct_var[-1] * 100

        # ── CVaR contribution ─────────────────────────────────────────────────
        pct_cvar = cvar_contribution(ret_aug, w_aug)
        qis_cvar = pct_cvar[-1] * 100

        # ── MDD contribution ──────────────────────────────────────────────────
        qis_mdd  = qis_mdd_contribution(base_r, blended_r) * 100

        records.append({
            "series":        series_name,
            "allocation_pct": round(alloc * 100, 0),
            "vol_contrib_pct":  round(qis_vol,  2),
            "var_contrib_pct":  round(qis_var,  2),
            "cvar_contrib_pct": round(qis_cvar, 2),
            "mdd_contrib_pct":  round(qis_mdd,  2),
        })

    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "vol_contrib_pct":  "Volatility % Contribution",
    "var_contrib_pct":  "VaR (95%) % Contribution",
    "cvar_contrib_pct": "CVaR (95%) % Contribution",
    "mdd_contrib_pct":  "Max Drawdown % Contribution",
}

SERIES_COLORS = {
    "qis_composite": "#1f77b4",
    "qis_momentum":  "#ff7f0e",
    "qis_value":     "#2ca02c",
    "qis_carry":     "#d62728",
    "qis_low_vol":   "#9467bd",
}


def plot_results(all_results: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "QIS Risk Contribution by Allocation Level\n"
        "(% of total portfolio risk, existing weights held fixed)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items()):
        for series in QIS_SERIES:
            subset = all_results[all_results["series"] == series]
            lw     = 2.5 if series == "qis_composite" else 1.4
            ls     = "-"  if series == "qis_composite" else "--"
            ax.plot(
                subset["allocation_pct"],
                subset[metric],
                label=series.replace("qis_", "").replace("_", " ").title(),
                color=SERIES_COLORS[series],
                linewidth=lw, linestyle=ls,
            )

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("QIS Allocation (%)")
        ax.set_ylabel("% Contribution to Portfolio Risk")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = Path(__file__).parent / "qis_risk_contribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    tickers, base_weights, port_ret, qis_ret, overlap = load_data()
    print(f"  Overlapping period : {overlap[0].date()} → {overlap[-1].date()}  ({len(overlap)} days)")
    print(f"  Portfolio tickers  : {tickers}")
    print(f"  Base portfolio sum : {base_weights.sum():.2f}\n")

    all_results = []

    for series in QIS_SERIES:
        print(f"Analysing {series} ...")
        df = analyse_qis_series(
            series_name  = series,
            tickers      = tickers,
            base_weights = base_weights,
            port_ret     = port_ret,
            qis_series   = qis_ret[series],
        )
        all_results.append(df)

        # print table for this series
        print(df.drop(columns="series").to_string(index=False))
        print()

    all_results = pd.concat(all_results, ignore_index=True)

    print("\n" + "="*70)
    print("SUMMARY: QIS Composite Risk Contribution")
    print("="*70)
    composite = all_results[all_results["series"] == "qis_composite"].drop(columns="series")
    print(composite.to_string(index=False))

    plot_results(all_results)


if __name__ == "__main__":
    main()
