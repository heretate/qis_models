"""
Value Factors
=============
Two signals based on mean-reversion to long-run equilibrium FX rates.
Both return a DataFrame of shape (dates × pairs).
Positive signal = foreign currency is undervalued vs USD (bullish foreign).

Both inputs are monthly series and are forward-filled to daily for consistency
with the other factor modules (though factor signals are only *used* at month-end
rebalance dates, so the filling does not introduce look-ahead bias).

Factors
-------
1. Effective Exchange Rate (NEER):
   Signal = historical_mean_NEER / current_NEER - 1
   Positive = current NEER is below its long-run mean → undervalued → bullish.

2. OECD PPP:
   Signal = historical_mean_PPP / current_PPP - 1
   PPP is expressed as USD per local currency unit.
   Positive = current PPP-implied rate is below long-run mean → undervalued.
"""

from __future__ import annotations

import pandas as pd

from fx_factor_analysis.config import G10_PAIRS, PAIR_FOREIGN


def _expanding_mean(series: pd.Series) -> pd.Series:
    """Expanding (full-history) mean up to each date. No look-ahead bias."""
    return series.expanding(min_periods=12).mean()  # require at least 12 obs (1 year monthly)


def effective_exchange_rate(data) -> pd.DataFrame:
    """
    Signal = expanding_mean(NEER) / current_NEER - 1

    NEER is a trade-weighted index: higher = stronger currency.
    If the currency has depreciated below its long-run average NEER,
    signal is positive (expected to revert = bullish).

    Monthly NEER is forward-filled to daily before computing the signal,
    so the last available monthly observation is used for intra-month dates.
    """
    neer = data.neer.copy()

    # Forward-fill to daily index using the spot index as reference
    daily_idx = data.spot.index
    neer = neer.reindex(daily_idx, method="ffill")

    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in neer.columns:
            continue
        s = neer[foreign]
        hist_mean = _expanding_mean(s)
        signals[pair] = hist_mean / s - 1.0

    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df


def ppp_value(data) -> pd.DataFrame:
    """
    Signal = expanding_mean(PPP) / current_PPP - 1

    PPP is the OECD implied exchange rate (USD per local currency unit).
    If current PPP is below its long-run mean, the foreign currency is
    cheap on a purchasing-power basis → positive signal (bullish foreign).

    Monthly PPP is forward-filled to daily.
    """
    ppp = data.ppp.copy()

    daily_idx = data.spot.index
    ppp = ppp.reindex(daily_idx, method="ffill")

    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in ppp.columns:
            continue
        s = ppp[foreign]
        hist_mean = _expanding_mean(s)
        signals[pair] = hist_mean / s - 1.0

    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df
