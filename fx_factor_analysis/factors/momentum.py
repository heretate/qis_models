"""
Momentum Factors
================
Four signals, all returning a DataFrame of shape (dates × pairs).
Positive signal = buy the foreign currency (go long foreign / short USD).

Factors
-------
1. Current vs HiLo  : (price - rolling_low) / (rolling_high - rolling_low)
2. Moving Average   : short_MA / long_MA - 1
3. Price Ranked     : % return over lookback (skip most recent month)
4. Skewness         : negative skewness of daily returns (negative = bullish)
"""

from __future__ import annotations

import pandas as pd

from fx_factor_analysis.config import G10_PAIRS, LookbackParams


def _spot(data) -> pd.DataFrame:
    """Convenience: spot prices for G10 pairs, columns in pair order."""
    return data.spot[[p for p in G10_PAIRS if p in data.spot.columns]]


def current_vs_hilo(data) -> pd.DataFrame:
    """
    Signal = (current_price - rolling_min) / (rolling_max - rolling_min)

    Range in [0, 1].  0 = at period low (bearish), 1 = at period high (bullish).
    Window: LookbackParams.HILO_WINDOW trading days (~52 weeks).
    """
    spot = _spot(data)
    w = LookbackParams.HILO_WINDOW
    lo = spot.rolling(w, min_periods=w // 2).min()
    hi = spot.rolling(w, min_periods=w // 2).max()
    signal = (spot - lo) / (hi - lo)
    signal.columns.name = "pair"
    return signal


def moving_average(data) -> pd.DataFrame:
    """
    Signal = short_MA / long_MA - 1

    Positive → price is above long-term trend (bullish momentum).
    Windows: LookbackParams.MA_SHORT (~1M) and MA_LONG (~12M).
    """
    spot = _spot(data)
    short_ma = spot.rolling(LookbackParams.MA_SHORT, min_periods=LookbackParams.MA_SHORT // 2).mean()
    long_ma = spot.rolling(LookbackParams.MA_LONG, min_periods=LookbackParams.MA_LONG // 2).mean()
    signal = short_ma / long_ma - 1.0
    signal.columns.name = "pair"
    return signal


def price_ranked(data) -> pd.DataFrame:
    """
    Signal = total log return from t-MOM_TOTAL to t-MOM_SKIP.

    Skips the most recent month to avoid short-term reversal contamination.
    Windows: LookbackParams.MOM_TOTAL (~12M), LookbackParams.MOM_SKIP (~1M).
    """
    spot = _spot(data)
    total = LookbackParams.MOM_TOTAL
    skip = LookbackParams.MOM_SKIP
    import numpy as np
    signal = np.log(spot.shift(skip) / spot.shift(total))
    signal.columns.name = "pair"
    return signal


def skewness(data) -> pd.DataFrame:
    """
    Signal = -skewness of daily log returns over the rolling window.

    Negative skewness in raw returns means left tail → negative expected carry
    from crash risk, so we *negate* to get a signal where higher = more bullish.
    Window: LookbackParams.SKEW_WINDOW (~3M).
    """
    spot = _spot(data)
    log_ret = spot.apply(lambda s: s.pct_change())  # or log returns — consistent with lit.
    w = LookbackParams.SKEW_WINDOW
    raw_skew = log_ret.rolling(w, min_periods=w // 2).skew()
    # Negate: more negative skewness → higher signal (crash risk premium)
    signal = -raw_skew
    signal.columns.name = "pair"
    return signal
