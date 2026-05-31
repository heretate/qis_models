# -*- coding: utf-8 -*-
"""
build_aggregate_returns.py
--------------------------
Computes daily aggregate portfolio and benchmark returns from 2020.

portfolio_return : buy-and-hold from inception - weights drift daily as
                   relative asset prices move.
benchmark_return : blended strategic benchmark using the same tickers and
                   target weights, rebalanced back to target at each
                   month-end (standard practice for strategic benchmarks).

Output columns:
  date | portfolio_return | benchmark_return |
  cumulative_return | benchmark_cumulative_return

Output: ../aggregate_portfolio_returns.csv (repo root)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUTPUT  = ROOT / "aggregate_portfolio_returns.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
holdings = pd.read_csv(ROOT / "portfolio_holdings.csv")
port_ret = (
    pd.read_csv(ROOT / "portfolio_returns.csv", parse_dates=["date"])
    .set_index("date")
    .sort_index()
)

tickers        = holdings["ticker"].tolist()
target_weights = holdings.set_index("ticker").loc[tickers, "weight"].values
ret            = port_ret[tickers].values   # (T, N)
dates          = port_ret.index
T, N           = ret.shape

# ── Portfolio return: buy-and-hold (weights drift with prices) ────────────────
port_daily = np.empty(T)
w = target_weights.copy().astype(float)
for t in range(T):
    port_daily[t] = w @ ret[t]
    w = w * (1 + ret[t])
    w /= w.sum()

# ── Benchmark return: monthly rebalance to target weights ─────────────────────
month_ends    = pd.DatetimeIndex(dates).to_period("M")
bench_daily   = np.empty(T)
w_b           = target_weights.copy().astype(float)
current_month = month_ends[0]

for t in range(T):
    if month_ends[t] != current_month:
        w_b           = target_weights.copy().astype(float)
        current_month = month_ends[t]
    bench_daily[t] = w_b @ ret[t]
    w_b = w_b * (1 + ret[t])
    w_b /= w_b.sum()

# ── Assemble output ───────────────────────────────────────────────────────────
agg = pd.DataFrame({
    "date":                        dates,
    "portfolio_return":            port_daily,
    "benchmark_return":            bench_daily,
    "cumulative_return":           (1 + port_daily).cumprod() - 1,
    "benchmark_cumulative_return": (1 + bench_daily).cumprod() - 1,
})

agg.to_csv(OUTPUT, index=False, float_format="%.8f")
print(f"Saved {len(agg)} rows -> {OUTPUT}")
print(agg.head(10).to_string(index=False))

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n-- Summary --")
for col, label in [("portfolio_return", "Portfolio"), ("benchmark_return", "Benchmark")]:
    ann_ret = agg[col].mean() * 252
    ann_vol = agg[col].std() * np.sqrt(252)
    print(f"  {label:12s}  Ann. Return: {ann_ret:.2%}  Ann. Vol: {ann_vol:.2%}")
