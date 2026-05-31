"""
Asset-Linked Factors
====================
Two signals linking FX to bond and equity markets.
Both return a DataFrame of shape (dates × pairs).
Positive signal = bullish for the foreign currency.

Factors
-------
1. Bond-Linked  : base_country_yield_change - quote_country_yield_change
   i.e. change in foreign 10Y yield - change in US 10Y yield
   Positive = foreign yields rising faster → capital inflow signal (carry-like)

   NOTE: sign convention can be debated. A rising yield might attract foreign
   capital (bullish FX) or signal deteriorating credit (bearish FX). The
   standard academic treatment for developed markets treats yield *rises* as
   USD-negative (safe-haven outflow), so we keep: foreign rise - USD rise > 0
   = bullish foreign.

2. Equity-Linked: base_country_equity_return - quote_country_equity_return
   i.e. foreign equity return - US equity return
   Positive = foreign equity outperforming → risk-on inflow into foreign currency.
"""

from __future__ import annotations

import pandas as pd

from fx_factor_analysis.config import G10_PAIRS, PAIR_FOREIGN


def bond_linked(data) -> pd.DataFrame:
    """
    Signal = Δ(foreign 10Y yield) - Δ(US 10Y yield)

    Daily first difference of 10Y yields.  Positive = foreign yields rising
    more than USD yields.
    """
    if "USD" not in data.yield_10y.columns:
        raise KeyError("USD 10Y yield not found.")
    usd_dy = data.yield_10y["USD"].diff()
    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in data.yield_10y.columns:
            continue
        foreign_dy = data.yield_10y[foreign].diff()
        signals[pair] = foreign_dy - usd_dy
    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df


def equity_linked(data) -> pd.DataFrame:
    """
    Signal = foreign equity daily return - US equity daily return

    Positive = foreign risk-on outperformance → bullish for foreign currency.
    Returns are simple % changes.
    """
    if "USD" not in data.equity.columns:
        raise KeyError("USD equity index not found.")
    usd_ret = data.equity["USD"].pct_change()
    signals = {}
    for pair in G10_PAIRS:
        foreign = PAIR_FOREIGN[pair]
        if foreign not in data.equity.columns:
            continue
        foreign_ret = data.equity[foreign].pct_change()
        signals[pair] = foreign_ret - usd_ret
    df = pd.DataFrame(signals)
    df.columns.name = "pair"
    return df
