"""
Factor Return Construction
==========================
Converts cross-sectional factor signals into long/short portfolio returns.

Methodology
-----------
1. Signals are evaluated at each month-end rebalance date.
2. Pairs are ranked cross-sectionally by signal value.
3. Top tercile  → long the foreign currency (long foreign / short USD)
4. Bottom tercile → short the foreign currency (short foreign / long USD)
5. Equal-weight within each leg.
6. Factor return = average return of long leg - average return of short leg,
   computed as the spot FX return from rebalance date t to t+1 (next month-end).
7. Spot FX returns are already in foreign/USD convention, so a positive spot
   return = the foreign currency appreciated = long foreign made money.

Public API
----------
    from fx_factor_analysis.construction.factor_returns import build_all_factor_returns
    factor_returns = build_all_factor_returns(data)   # returns pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from fx_factor_analysis.config import G10_PAIRS, PortfolioParams
from fx_factor_analysis.data.bbg_fetcher import FXFactorData

# Factor modules
from fx_factor_analysis.factors import (
    asset_linked,
    carry,
    momentum,
    value,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal catalogue
# Factor name → callable(data) → signal DataFrame (dates × pairs)
# ---------------------------------------------------------------------------

FACTOR_SIGNAL_FUNCTIONS: dict[str, Callable[[FXFactorData], pd.DataFrame]] = {
    # Momentum
    "mom_hilo":       momentum.current_vs_hilo,
    "mom_ma":         momentum.moving_average,
    "mom_price":      momentum.price_ranked,
    "mom_skew":       momentum.skewness,
    # Carry
    "carry_yc":       carry.yield_curve_steepness,
    "carry_fwd":      carry.forward_curve,
    "carry_rate":     carry.interest_differential,
    # Asset-linked
    "asset_bond":     asset_linked.bond_linked,
    "asset_equity":   asset_linked.equity_linked,
    # Value
    "value_neer":     value.effective_exchange_rate,
    "value_ppp":      value.ppp_value,
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return business-month-end dates that exist within the data index."""
    freq = PortfolioParams.REBALANCE_FREQ          # "BME"
    candidate = pd.date_range(index.min(), index.max(), freq=freq)
    # Snap to the nearest available date in the actual index (in case BME falls on holiday)
    snapped = [index[index.get_indexer([d], method="ffill")[0]] for d in candidate]
    return pd.DatetimeIndex(sorted(set(snapped)))


def _spot_monthly_returns(spot: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute FX spot returns between consecutive rebalance dates.
    Return[t] = spot[t+1] / spot[t] - 1, indexed at rebalance date t.
    (i.e. the return earned by holding foreign currency from t to t+1)
    """
    # Spot at each rebalance date
    spot_at_rebal = spot.reindex(rebal_dates, method="ffill")
    # Shift back: ret at date[i] = price[i+1] / price[i] - 1
    fwd_ret = spot_at_rebal.shift(-1) / spot_at_rebal - 1.0
    return fwd_ret  # NaN at the last date (no next period)


def _rank_signal(signal_row: pd.Series, n: int) -> pd.Series:
    """
    Cross-sectionally rank a signal at a single date.
    Returns a position Series: +1 (long), -1 (short), 0 (neutral).
    Top n → long, bottom n → short.
    """
    valid = signal_row.dropna()
    if len(valid) < 2 * n:
        return pd.Series(0.0, index=signal_row.index)
    ranks = valid.rank(ascending=True)
    positions = pd.Series(0.0, index=signal_row.index)
    n_pairs = len(valid)
    positions[ranks <= n] = -1.0             # bottom tercile: short
    positions[ranks > n_pairs - n] = 1.0     # top tercile: long
    return positions


def _compute_factor_return(
    signal: pd.DataFrame,
    spot_monthly_ret: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    For a single factor signal:
    1. Sample signal at each rebalance date.
    2. Rank → positions.
    3. Multiply by next-period spot returns.
    4. Average long leg - average short leg.
    """
    n = PortfolioParams.TERCILE_N
    returns = []
    dates = []

    for date in rebal_dates[:-1]:  # last date has no forward return
        if date not in signal.index:
            # Find nearest prior signal date
            prior = signal.index[signal.index <= date]
            if prior.empty:
                continue
            date_sig = prior[-1]
        else:
            date_sig = date

        sig_row = signal.loc[date_sig]
        pos = _rank_signal(sig_row, n)

        # Forward return for this period
        if date not in spot_monthly_ret.index:
            continue
        fwd = spot_monthly_ret.loc[date]

        long_mask = pos == 1.0
        short_mask = pos == -1.0

        long_ret = fwd[long_mask].mean() if long_mask.any() else np.nan
        short_ret = fwd[short_mask].mean() if short_mask.any() else np.nan

        if np.isnan(long_ret) or np.isnan(short_ret):
            factor_ret = np.nan
        else:
            factor_ret = long_ret - short_ret

        returns.append(factor_ret)
        dates.append(date)

    return pd.Series(returns, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_all_factor_returns(data: FXFactorData) -> pd.DataFrame:
    """
    Compute return series for all 11 FX factors.

    Parameters
    ----------
    data : FXFactorData
        Raw Bloomberg data from BBGFetcher.fetch_all().

    Returns
    -------
    pd.DataFrame
        Monthly factor returns.  Index = rebalance dates (business month-end).
        Columns = factor names (see FACTOR_SIGNAL_FUNCTIONS keys).
        Values = long-minus-short portfolio return for that month.
    """
    pairs = [p for p in G10_PAIRS if p in data.spot.columns]
    spot = data.spot[pairs]

    rebal_dates = _rebalance_dates(spot.index)
    spot_monthly_ret = _spot_monthly_returns(spot, rebal_dates)

    all_returns: dict[str, pd.Series] = {}

    for name, fn in FACTOR_SIGNAL_FUNCTIONS.items():
        logger.info("Computing factor: %s", name)
        try:
            signal = fn(data)
            # Restrict to our pair universe
            signal = signal[[c for c in pairs if c in signal.columns]]
            factor_ret = _compute_factor_return(signal, spot_monthly_ret, rebal_dates)
            all_returns[name] = factor_ret
        except Exception as e:
            logger.error("Failed to compute factor %s: %s", name, e)
            all_returns[name] = pd.Series(dtype=float)

    factor_returns = pd.DataFrame(all_returns)
    factor_returns.index.name = "date"
    factor_returns = factor_returns.sort_index()

    logger.info(
        "Factor returns computed: %d periods, %d factors.",
        len(factor_returns),
        factor_returns.shape[1],
    )
    return factor_returns
