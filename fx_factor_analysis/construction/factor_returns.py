"""
Factor Return Construction
==========================
Converts cross-sectional factor signals into daily long/short portfolio returns
with monthly rebalancing.

Methodology
-----------
1. Signals are evaluated at each month-end rebalance date.
2. Pairs are ranked cross-sectionally by signal value.
3. Top tercile  → long the foreign currency (long foreign / short USD)
4. Bottom tercile → short the foreign currency (short foreign / long USD)
5. Equal-weight within each leg; positions are constant between rebalance dates.
6. Daily factor return = weighted sum of daily spot FX returns using the
   positions set at the most recent month-end.
   - Long leg contributes +1/n * daily_spot_return
   - Short leg contributes -1/n * daily_spot_return
7. Spot FX returns are in foreign/USD convention (positive = foreign appreciated).

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


def _daily_spot_returns(spot: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns for each pair. Return[t] = spot[t] / spot[t-1] - 1."""
    return spot.pct_change()


def _build_position_frame(
    signal: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    daily_index: pd.DatetimeIndex,
    n: int,
) -> pd.DataFrame:
    """
    Build a daily position DataFrame by:
    1. Computing positions at each rebalance date from the signal.
    2. Forward-filling those positions to cover every business day until
       the next rebalance.

    Positions are normalised so each leg sums to ±1 in absolute weight
    (i.e. +1/n per long pair, -1/n per short pair), giving a zero-cost
    long/short portfolio.

    Returns
    -------
    pd.DataFrame
        Shape (daily_index × pairs).  Values ∈ {-1/n, 0, +1/n}.
        NaN where no signal was available.
    """
    # --- Compute positions at each rebalance date ---
    pos_at_rebal = pd.DataFrame(index=rebal_dates, columns=signal.columns, dtype=float)

    for date in rebal_dates:
        # Use the most recent available signal on or before this rebalance date
        prior = signal.index[signal.index <= date]
        if prior.empty:
            pos_at_rebal.loc[date] = np.nan
            continue
        sig_row = signal.loc[prior[-1]]
        raw_pos = _rank_signal(sig_row, n)
        # Normalise: divide by n so each leg's total absolute weight = 1
        pos_at_rebal.loc[date] = raw_pos / n

    # --- Forward-fill to daily frequency ---
    # Reindex to the full daily index; ffill carries each rebalance position
    # forward until the next month-end.
    pos_daily = pos_at_rebal.reindex(daily_index, method="ffill")
    return pos_daily


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
    daily_spot_ret: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    For a single factor:
    1. Build daily position frame from monthly rebalance signals.
    2. Daily factor return = sum(position_i * daily_spot_return_i) across all pairs.
       Since positions are ±1/n, this equals:
         mean(daily_ret of long pairs) - mean(daily_ret of short pairs)

    The first day in each new holding period (the rebalance date itself) is
    treated as a transition day: positions from the *previous* month are used
    for that day's return, as the new signal is set at the close.
    """
    n = PortfolioParams.TERCILE_N
    daily_index = daily_spot_ret.index

    pos_daily = _build_position_frame(signal, rebal_dates, daily_index, n)

    # Align positions and returns to the same columns
    common_pairs = pos_daily.columns.intersection(daily_spot_ret.columns)
    pos = pos_daily[common_pairs]
    ret = daily_spot_ret[common_pairs]

    # Daily factor return: dot product of position vector and return vector
    # (NaN pairs are excluded automatically via fillna(0) on positions)
    factor_ret = (pos.fillna(0.0) * ret).sum(axis=1)

    # Set to NaN on days where we had no valid position at all
    no_position = pos.fillna(0.0).abs().sum(axis=1) == 0
    factor_ret[no_position] = np.nan

    return factor_ret


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_all_factor_returns(data: FXFactorData) -> pd.DataFrame:
    """
    Compute daily return series for all 11 FX factors with monthly rebalancing.

    Parameters
    ----------
    data : FXFactorData
        Raw Bloomberg data from BBGFetcher.fetch_all().

    Returns
    -------
    pd.DataFrame
        Daily factor returns.  Index = all business days in the data.
        Columns = factor names (see FACTOR_SIGNAL_FUNCTIONS keys).
        Values = daily long-minus-short portfolio return.
    """
    pairs = [p for p in G10_PAIRS if p in data.spot.columns]
    spot = data.spot[pairs]

    rebal_dates = _rebalance_dates(spot.index)
    daily_spot_ret = _daily_spot_returns(spot)

    all_returns: dict[str, pd.Series] = {}

    for name, fn in FACTOR_SIGNAL_FUNCTIONS.items():
        logger.info("Computing factor: %s", name)
        try:
            signal = fn(data)
            signal = signal[[c for c in pairs if c in signal.columns]]
            factor_ret = _compute_factor_return(signal, daily_spot_ret, rebal_dates)
            all_returns[name] = factor_ret
        except Exception as e:
            logger.error("Failed to compute factor %s: %s", name, e)
            all_returns[name] = pd.Series(dtype=float)

    factor_returns = pd.DataFrame(all_returns)
    factor_returns.index.name = "date"
    factor_returns = factor_returns.sort_index()

    logger.info(
        "Factor returns computed: %d daily observations, %d factors.",
        len(factor_returns),
        factor_returns.shape[1],
    )
    return factor_returns
