# -*- coding: utf-8 -*-
"""
portfolio_beta.py
-----------------
Computes rolling 1-year (252-day) OLS beta of the portfolio against
the blended benchmark using daily returns, then plots beta over time.

Input  : ../aggregate_portfolio_returns.csv
Output : portfolio_beta.png (same folder as this script)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUTPUT  = Path(__file__).parent / "portfolio_beta.png"
WINDOW  = 252   # 1 trading year

# ── Load ──────────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(ROOT / "aggregate_portfolio_returns.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

y = df["portfolio_return"].values   # dependent
x = df["benchmark_return"].values   # independent
dates = df.index
T = len(y)

# ── Rolling OLS beta ──────────────────────────────────────────────────────────
# beta = cov(y,x) / var(x)  over the rolling window
betas  = np.full(T, np.nan)
alphas = np.full(T, np.nan)
r2s    = np.full(T, np.nan)

for t in range(WINDOW - 1, T):
    y_w = y[t - WINDOW + 1 : t + 1]
    x_w = x[t - WINDOW + 1 : t + 1]

    x_dm = x_w - x_w.mean()
    y_dm = y_w - y_w.mean()

    var_x = (x_dm ** 2).mean()
    if var_x == 0:
        continue

    beta  = (x_dm * y_dm).mean() / var_x
    alpha = y_w.mean() - beta * x_w.mean()

    y_hat = alpha + beta * x_w
    ss_res = ((y_w - y_hat) ** 2).sum()
    ss_tot = ((y_w - y_w.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    betas[t]  = beta
    alphas[t] = alpha * 252          # annualise alpha
    r2s[t]    = r2

valid = ~np.isnan(betas)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    "Portfolio vs Benchmark: Rolling 1-Year OLS Regression (Daily Returns)",
    fontsize=13, fontweight="bold"
)

# --- Beta ---
ax = axes[0]
ax.plot(dates[valid], betas[valid], color="#1f77b4", linewidth=1.5, label="Rolling Beta")
ax.axhline(1.0, color="black",  linewidth=0.8, linestyle="--", label="Beta = 1")
ax.axhline(np.nanmean(betas), color="#ff7f0e", linewidth=1.0,
           linestyle=":", label=f"Mean Beta ({np.nanmean(betas):.3f})")
ax.fill_between(dates[valid], betas[valid], 1.0,
                where=betas[valid] > 1.0, alpha=0.15, color="#d62728", label="Beta > 1")
ax.fill_between(dates[valid], betas[valid], 1.0,
                where=betas[valid] < 1.0, alpha=0.15, color="#2ca02c", label="Beta < 1")
ax.set_ylabel("Beta")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# --- Annualised Alpha ---
ax = axes[1]
ax.plot(dates[valid], alphas[valid] * 100, color="#2ca02c", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.fill_between(dates[valid], alphas[valid] * 100, 0,
                where=alphas[valid] >= 0, alpha=0.2, color="#2ca02c")
ax.fill_between(dates[valid], alphas[valid] * 100, 0,
                where=alphas[valid] < 0,  alpha=0.2, color="#d62728")
ax.set_ylabel("Annualised Alpha (%)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.grid(True, alpha=0.3)

# --- R-squared ---
ax = axes[2]
ax.plot(dates[valid], r2s[valid], color="#9467bd", linewidth=1.5)
ax.axhline(np.nanmean(r2s), color="#ff7f0e", linewidth=1.0,
           linestyle=":", label=f"Mean R2 ({np.nanmean(r2s):.3f})")
ax.set_ylabel("R-squared")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())

fig.autofmt_xdate(rotation=0, ha="center")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n-- Rolling Beta Summary (window={WINDOW}d) --")
print(f"  Mean beta   : {np.nanmean(betas):.4f}")
print(f"  Std beta    : {np.nanstd(betas):.4f}")
print(f"  Min beta    : {np.nanmin(betas):.4f}  ({dates[np.nanargmin(betas)].date()})")
print(f"  Max beta    : {np.nanmax(betas):.4f}  ({dates[np.nanargmax(betas)].date()})")
print(f"  Mean R2     : {np.nanmean(r2s):.4f}")
print(f"  Mean ann. alpha : {np.nanmean(alphas)*100:.2f}%")
