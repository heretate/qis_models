"""
Carry Factors
=============
Three signals, all returning a DataFrame of shape (dates × pairs).
Positive signal = high carry for the foreign currency (go long foreign / short USD).

Factors
-------
1. Yield Curve Steepness : (10Y - 2Y) of base - (10Y - 2Y) of quote
   Base  = foreign currency country
   Quote = USD
   Steeper foreign YC → higher term premium → bullish for foreign currency

2. Forward Curve         : (spot - 1M_forward) / spot
   Positive = forward discount of foreign (USD trades at forward premium)
   = classic CIP-based carry

3. Interest Differential : short_rate_foreign - short_rate_USD
"""

from __future__ import annotations

import pandas as pd

from fx_factor_analysis.config import G10_PAIRS, PAIR_FOREIGN


def _yc_steepness(yield_10y: pd.DataFrame, yield_2y: pd.DataFrame, ccy: str) -> pd.Series:
    """10Y - 2Y for a given currency, returning a Series aligned to the index."""
    return yield_10y[ccy] - yield_2y[ccy]


def yield_curve_steepness(data) -> pd.DataFrame:
    """
    Signal = steepness_foreign - steepness_USD
    steepness = 10Y yield - 2Y yield  (basis points or %)

    Higher relative steepness of the foreign yield curve → bullish signal.
    """
    usd_steepness = _yc_steepness(data.yield_10y, data.yield_2y, "USD")
    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in data.yield_10y.columns or foreign not in data.yield_2y.columns:
            continue
        foreign_steepness = _yc_steepness(data.yield_10y, data.yield_2y, foreign)
        signals[pair] = foreign_steepness - usd_steepness
    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df


def forward_curve(data) -> pd.DataFrame:
    """
    Signal = (spot - forward_1m) / spot

    Positive → foreign currency trades at a forward discount to USD
    (i.e. you earn carry by holding the foreign currency).
    Both spot and forward are in foreign/USD convention (from bbg_fetcher).
    """
    signals = {}
    for pair in G10_PAIRS:
        if pair not in data.spot.columns or pair not in data.forward_1m.columns:
            continue
        s = data.spot[pair]
        f = data.forward_1m[pair]
        signals[pair] = (s - f) / s
    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df


def interest_differential(data) -> pd.DataFrame:
    """
    Signal = short_rate_foreign - short_rate_USD

    Higher foreign rate → positive carry (classic uncovered interest parity carry).
    Rates in % annualised.
    """
    if "USD" not in data.short_rate.columns:
        raise KeyError("USD short rate not found in data.short_rate.")
    usd_rate = data.short_rate["USD"]
    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in data.short_rate.columns:
            continue
        signals[pair] = data.short_rate[foreign] - usd_rate
    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df
