"""
Bloomberg Data Fetcher
======================
Wraps blpapi to pull historical and reference data needed for FX factor construction.

Usage
-----
    fetcher = BBGFetcher()
    data = fetcher.fetch_all(start_date="2005-01-01", end_date="2024-12-31")
    fetcher.close()

    # Or use as context manager:
    with BBGFetcher() as fetcher:
        data = fetcher.fetch_all(start_date="2005-01-01", end_date="2024-12-31")

Returns a FXFactorData dataclass containing one DataFrame per data type.
All DataFrames are indexed by DatetimeIndex, columns are currency codes or pair names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import blpapi
import pandas as pd

from fx_factor_analysis.config import (
    EQUITY_TICKERS,
    FORWARD_1M_TICKERS,
    G10_PAIRS,
    NEER_TICKERS,
    PAIR_FOREIGN,
    PPP_TICKERS,
    SHORT_RATE_TICKERS,
    SPOT_TICKERS,
    YIELD_10Y_TICKERS,
    YIELD_2Y_TICKERS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FXFactorData:
    """
    All raw data needed by the factor modules.
    Each DataFrame: DatetimeIndex rows, named columns (pair or currency code).
    """
    # Spot FX rates (all normalised to foreign/USD, i.e. USD per 1 unit of foreign)
    spot: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 1M outright forward rates (same quoting convention as spot)
    forward_1m: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 3M OIS rates (%, annualised) — columns = currency codes incl. USD
    short_rate: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 2Y government bond yields (%) — columns = currency codes incl. USD
    yield_2y: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 10Y government bond yields (%) — columns = currency codes incl. USD
    yield_10y: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Equity index levels — columns = currency codes incl. USD
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)

    # BIS NEER indices (monthly) — columns = currency codes
    neer: pd.DataFrame = field(default_factory=pd.DataFrame)

    # OECD PPP rates (monthly) — columns = currency codes (no USD; USD is base)
    ppp: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class BBGFetcher:
    """
    Thin blpapi wrapper.  Handles session lifecycle, BDH bulk pulls, and
    normalises output into tidy DataFrames.

    Parameters
    ----------
    host : str
        Bloomberg server host (default "localhost" for desktop Terminal).
    port : int
        Bloomberg server port (default 8194).
    """

    _BBG_DATE_FMT = "%Y%m%d"

    def __init__(self, host: str = "localhost", port: int = 8194) -> None:
        self._host = host
        self._port = port
        self._session: Optional[blpapi.Session] = None
        self._refdata_service: Optional[blpapi.Service] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open Bloomberg session and start the refdata service."""
        opts = blpapi.SessionOptions()
        opts.setServerHost(self._host)
        opts.setServerPort(self._port)
        self._session = blpapi.Session(opts)
        if not self._session.start():
            raise ConnectionError("Failed to start Bloomberg session.")
        if not self._session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdata service.")
        self._refdata_service = self._session.getService("//blp/refdata")
        logger.info("Bloomberg session opened.")

    def close(self) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None
            logger.info("Bloomberg session closed.")

    def __enter__(self) -> "BBGFetcher":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core BDH (historical data) pull
    # ------------------------------------------------------------------

    def bdh(
        self,
        tickers: list[str],
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
        currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pull historical data (BDH) for a list of tickers and fields.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (ticker, field).  DatetimeIndex rows.
            Missing dates/tickers have NaN.
        """
        if self._session is None:
            raise RuntimeError("Session not open. Call open() first.")

        request = self._refdata_service.createRequest("HistoricalDataRequest")
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in fields:
            request.getElement("fields").appendValue(field)
        request.set("startDt", start_date.replace("-", ""))
        request.set("endDt", end_date.replace("-", ""))
        request.set("periodicitySelection", periodicity)
        if currency:
            request.set("currency", currency)

        self._session.sendRequest(request)

        records: list[dict] = []
        done = False
        while not done:
            event = self._session.nextEvent(timeout=10_000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    ticker_val = sec_data.getElementAsString("security")
                    field_data_arr = sec_data.getElement("fieldData")
                    for i in range(field_data_arr.numValues()):
                        fd = field_data_arr.getValueAsElement(i)
                        row: dict = {"date": fd.getElementAsDatetime("date"), "ticker": ticker_val}
                        for f in fields:
                            row[f] = fd.getElementAsFloat(f) if fd.hasElement(f) else float("nan")
                        records.append(row)
            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        if not records:
            logger.warning("BDH returned no data for tickers: %s", tickers)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "ticker"])
        # Pivot to wide: index=date, columns=ticker (single field case)
        if len(fields) == 1:
            df = df[fields[0]].unstack("ticker")
        else:
            df = df.unstack("ticker")
        df.index.name = "date"
        df.index = pd.DatetimeIndex(df.index)
        return df

    def bdh_single_field(
        self,
        ticker_map: dict[str, str],
        field: str,
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
    ) -> pd.DataFrame:
        """
        Convenience wrapper: pull one field for a {label: bbg_ticker} mapping.
        Returns DataFrame with columns = labels (not raw Bloomberg tickers).

        Parameters
        ----------
        ticker_map : dict[str, str]
            {column_label: bloomberg_ticker}, e.g. {"EURUSD": "EURUSD Curncy"}
        """
        bbg_tickers = list(ticker_map.values())
        raw = self.bdh(bbg_tickers, [field], start_date, end_date, periodicity)
        if raw.empty:
            return pd.DataFrame()
        # Rename columns from BBG ticker → label
        reverse_map = {v: k for k, v in ticker_map.items()}
        raw = raw.rename(columns=reverse_map)
        # Keep only columns we asked for (drop any extras from BBG response)
        cols = [c for c in ticker_map.keys() if c in raw.columns]
        return raw[cols]

    # ------------------------------------------------------------------
    # High-level fetch_all
    # ------------------------------------------------------------------

    def fetch_all(self, start_date: str, end_date: str) -> FXFactorData:
        """
        Pull all data required for FX factor construction.

        Parameters
        ----------
        start_date, end_date : str
            ISO format "YYYY-MM-DD".  Add extra history buffer before calling
            if you need warm-up data for rolling windows (e.g. start_date 1 year
            earlier than your actual analysis start).

        Returns
        -------
        FXFactorData
        """
        logger.info("Fetching Bloomberg data: %s → %s", start_date, end_date)
        data = FXFactorData()

        # --- Spot FX (daily) ---
        logger.info("  Spot FX...")
        spot_raw = self.bdh_single_field(SPOT_TICKERS, "PX_LAST", start_date, end_date, "DAILY")
        data.spot = self._normalise_spot(spot_raw)

        # --- 1M Forward (daily) ---
        logger.info("  1M Forwards...")
        fwd_raw = self.bdh_single_field(FORWARD_1M_TICKERS, "PX_LAST", start_date, end_date, "DAILY")
        data.forward_1m = self._normalise_spot(fwd_raw)  # same quoting inversion logic

        # --- Short rates / OIS (daily) ---
        logger.info("  Short rates (OIS 3M)...")
        data.short_rate = self.bdh_single_field(
            SHORT_RATE_TICKERS, "PX_LAST", start_date, end_date, "DAILY"
        )

        # --- Yield curve: 2Y and 10Y (daily) ---
        logger.info("  Government bond yields (2Y, 10Y)...")
        data.yield_2y = self.bdh_single_field(
            YIELD_2Y_TICKERS, "PX_LAST", start_date, end_date, "DAILY"
        )
        data.yield_10y = self.bdh_single_field(
            YIELD_10Y_TICKERS, "PX_LAST", start_date, end_date, "DAILY"
        )

        # --- Equity indices (daily) ---
        logger.info("  Equity indices...")
        data.equity = self.bdh_single_field(
            EQUITY_TICKERS, "PX_LAST", start_date, end_date, "DAILY"
        )

        # --- NEER (monthly) ---
        logger.info("  NEER (monthly)...")
        data.neer = self.bdh_single_field(
            NEER_TICKERS, "PX_LAST", start_date, end_date, "MONTHLY"
        )

        # --- PPP (monthly) ---
        logger.info("  PPP (monthly)...")
        data.ppp = self.bdh_single_field(
            PPP_TICKERS, "PX_LAST", start_date, end_date, "MONTHLY"
        )

        logger.info("Data fetch complete.")
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_spot(df: pd.DataFrame) -> pd.DataFrame:
        """
        Invert USD-base pairs so all columns express foreign/USD
        (i.e. units of USD you receive per 1 unit of foreign currency).

        e.g. USDJPY = 150 → JPYUSD = 1/150 ≈ 0.00667
        The column is still labelled by the original pair name for traceability,
        but callers should use PAIR_FOREIGN to identify the foreign leg.
        """
        from fx_factor_analysis.config import USD_BASE_PAIRS

        df = df.copy()
        for pair in USD_BASE_PAIRS:
            if pair in df.columns:
                df[pair] = 1.0 / df[pair]
        return df
