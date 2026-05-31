"""
FX Factor Analysis — Configuration
====================================
All parameters, universe definitions, and Bloomberg ticker mappings in one place.
Edit here; no changes needed elsewhere for routine parameter tweaks.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

# G10 pairs, all expressed as XXX/USD (i.e. units of USD per 1 unit of foreign).
# USDJPY, USDCAD, USDCHF, USDSEK, USDNOK are inverted from standard quoting
# to keep a consistent "USD per foreign" convention across the board.
G10_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "NZDUSD",
    "USDJPY",   # inverted → JPYUSD internally
    "USDCAD",   # inverted → CADUSD internally
    "USDCHF",   # inverted → CHFUSD internally
    "USDSEK",   # inverted → SEKUSD internally
    "USDNOK",   # inverted → NOKUSD internally
]

# Pairs where the Bloomberg ticker quotes USD as the BASE (USD/foreign).
# These are inverted so every pair becomes "foreign per USD" → positive = USD weak.
USD_BASE_PAIRS = {"USDJPY", "USDCAD", "USDCHF", "USDSEK", "USDNOK"}

# 2-letter country codes for each foreign currency (used to look up rates/equities)
CURRENCY_COUNTRY: dict[str, str] = {
    "EUR": "EU",
    "GBP": "UK",
    "AUD": "AU",
    "NZD": "NZ",
    "JPY": "JP",
    "CAD": "CA",
    "CHF": "CH",
    "SEK": "SE",
    "NOK": "NO",
    "USD": "US",
}

# Foreign currency for each pair (after normalising to foreign/USD convention)
PAIR_FOREIGN: dict[str, str] = {
    "EURUSD": "EUR",
    "GBPUSD": "GBP",
    "AUDUSD": "AUD",
    "NZDUSD": "NZD",
    "USDJPY": "JPY",
    "USDCAD": "CAD",
    "USDCHF": "CHF",
    "USDSEK": "SEK",
    "USDNOK": "NOK",
}


# ---------------------------------------------------------------------------
# Bloomberg Ticker Maps
# ---------------------------------------------------------------------------

# Spot FX tickers (Bloomberg Curncy)
SPOT_TICKERS: dict[str, str] = {
    "EURUSD": "EURUSD Curncy",
    "GBPUSD": "GBPUSD Curncy",
    "AUDUSD": "AUDUSD Curncy",
    "NZDUSD": "NZDUSD Curncy",
    "USDJPY": "USDJPY Curncy",
    "USDCAD": "USDCAD Curncy",
    "USDCHF": "USDCHF Curncy",
    "USDSEK": "USDSEK Curncy",
    "USDNOK": "USDNOK Curncy",
}

# 1-month outright forward tickers
FORWARD_1M_TICKERS: dict[str, str] = {
    "EURUSD": "EURUS1M BGN Curncy",
    "GBPUSD": "GBPUS1M BGN Curncy",
    "AUDUSD": "AUDUS1M BGN Curncy",
    "NZDUSD": "NZDUS1M BGN Curncy",
    "USDJPY": "USDYJ1M BGN Curncy",
    "USDCAD": "USDCA1M BGN Curncy",
    "USDCHF": "USDFS1M BGN Curncy",
    "USDSEK": "USDSK1M BGN Curncy",
    "USDNOK": "USDNK1M BGN Curncy",
}

# 3-month OIS (overnight index swap) rates as short-end proxy for each currency
# Used for: Carry Interest Differential
SHORT_RATE_TICKERS: dict[str, str] = {
    "EUR": "EUSWEC Curncy",   # EUR OIS 3M
    "GBP": "BPSWSC Curncy",   # GBP OIS 3M
    "AUD": "ADSWSC Curncy",   # AUD OIS 3M
    "NZD": "NDSWSC Curncy",   # NZD OIS 3M
    "JPY": "JYSWSC Curncy",   # JPY OIS 3M
    "CAD": "CDSWSC Curncy",   # CAD OIS 3M
    "CHF": "SFSWSC Curncy",   # CHF OIS 3M
    "SEK": "SKSWSC Curncy",   # SEK OIS 3M
    "NOK": "NKSWSC Curncy",   # NOK OIS 3M
    "USD": "USOSFR3 Curncy",  # USD OIS 3M
}

# 2-year government bond yields (for yield curve steepness & bond-linked signal)
YIELD_2Y_TICKERS: dict[str, str] = {
    "EUR": "GDBR2 Index",    # Germany 2Y Bund
    "GBP": "GUKG2 Index",    # UK Gilt 2Y
    "AUD": "GACGB2 Index",   # Australia 2Y
    "NZD": "GNZGB2 Index",   # New Zealand 2Y
    "JPY": "GJGB2 Index",    # Japan JGB 2Y
    "CAD": "GCAN2YR Index",  # Canada 2Y
    "CHF": "GSWISS2 Index",  # Switzerland 2Y
    "SEK": "GSWE2YR Index",  # Sweden 2Y
    "NOK": "GNOR2YR Index",  # Norway 2Y
    "USD": "USGG2YR Index",  # US Treasury 2Y
}

# 10-year government bond yields
YIELD_10Y_TICKERS: dict[str, str] = {
    "EUR": "GDBR10 Index",    # Germany 10Y Bund
    "GBP": "GUKG10 Index",    # UK Gilt 10Y
    "AUD": "GACGB10 Index",   # Australia 10Y
    "NZD": "GNZGB10 Index",   # New Zealand 10Y
    "JPY": "GJGB10 Index",    # Japan JGB 10Y
    "CAD": "GCAN10YR Index",  # Canada 10Y
    "CHF": "GSWISS10 Index",  # Switzerland 10Y
    "SEK": "GSWE10YR Index",  # Sweden 10Y
    "NOK": "GNOR10YR Index",  # Norway 10Y
    "USD": "USGG10YR Index",  # US Treasury 10Y
}

# Broad equity indices per country
EQUITY_TICKERS: dict[str, str] = {
    "EUR": "SX5E Index",   # Euro Stoxx 50
    "GBP": "UKX Index",    # FTSE 100
    "AUD": "AS51 Index",   # ASX 200
    "NZD": "NZSE50FG Index",  # NZX 50
    "JPY": "NKY Index",    # Nikkei 225
    "CAD": "SPTSX Index",  # S&P/TSX Composite
    "CHF": "SMI Index",    # SMI
    "SEK": "OMX Index",    # OMX Stockholm 30
    "NOK": "OBX Index",    # OBX
    "USD": "SPX Index",    # S&P 500
}

# BIS Nominal Effective Exchange Rate (NEER) indices
# Monthly series. Field: PX_LAST
NEER_TICKERS: dict[str, str] = {
    "EUR": "BISNEEUR Index",
    "GBP": "BISNEEGB Index",
    "AUD": "BISNEEAUD Index",
    "NZD": "BISNZENZER Index",
    "JPY": "BISNEEJP Index",
    "CAD": "BISNEECA Index",
    "CHF": "BISNEECHE Index",
    "SEK": "BISNEESWK Index",
    "NOK": "BISNOENOK Index",
    "USD": "BISNEEUS Index",
}

# OECD PPP (USD per local currency unit, monthly)
PPP_TICKERS: dict[str, str] = {
    "EUR": "OECDPPEU Index",
    "GBP": "OECDPPGB Index",
    "AUD": "OECDPPAU Index",
    "NZD": "OECDPPNZ Index",
    "JPY": "OECDPPJP Index",
    "CAD": "OECDPPCA Index",
    "CHF": "OECDPPCH Index",
    "SEK": "OECDPPSE Index",
    "NOK": "OECDPPNO Index",
}


# ---------------------------------------------------------------------------
# Lookback / Window Parameters
# ---------------------------------------------------------------------------

class LookbackParams:
    # Momentum: Current vs HiLo
    HILO_WINDOW: int = 260          # ~52 weeks in trading days

    # Momentum: Moving Average
    MA_SHORT: int = 21              # ~1 month
    MA_LONG: int = 252              # ~12 months

    # Momentum: Price Ranked (skip last month to avoid reversal)
    MOM_TOTAL: int = 252            # 12-month window
    MOM_SKIP: int = 21              # skip most recent month

    # Momentum: Skewness
    SKEW_WINDOW: int = 63           # ~3 months

    # Value: NEER/PPP — use full available history as the "historical average"
    # No explicit window; computed over all available data up to signal date.

    # Yield curve steepness = 10Y - 2Y (tenors defined in tickers above)


# ---------------------------------------------------------------------------
# Factor Return Construction
# ---------------------------------------------------------------------------

class PortfolioParams:
    REBALANCE_FREQ: str = "BME"     # pandas BusinessMonthEnd offset alias
    TERCILE_N: int = 3              # top/bottom N pairs per leg (9 pairs → 3 per tercile)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = "output"
FACTOR_RETURNS_CSV: str = "factor_returns.csv"
FACTOR_RETURNS_PARQUET: str = "factor_returns.parquet"
