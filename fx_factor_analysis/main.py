"""
FX Factor Return Construction — Orchestrator
=============================================
Entry point.  Pulls Bloomberg data, computes all 11 factor signals,
constructs monthly long/short returns, and writes output files.

Usage
-----
    python -m fx_factor_analysis.main --start 2005-01-01 --end 2024-12-31

    # With extra warm-up buffer (recommended — rolling windows need history):
    python -m fx_factor_analysis.main --start 2005-01-01 --end 2024-12-31 --warmup 260

Output files (written to OUTPUT_DIR defined in config.py):
    factor_returns.csv
    factor_returns.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd

# Configure logging before importing project modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FX factor return series from Bloomberg data."
    )
    parser.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD",
        help="Analysis start date (factor returns will begin here).",
    )
    parser.add_argument(
        "--end", default=date.today().isoformat(), metavar="YYYY-MM-DD",
        help="Analysis end date. Defaults to today.",
    )
    parser.add_argument(
        "--warmup", type=int, default=260, metavar="DAYS",
        help=(
            "Number of extra calendar days to pull *before* --start to warm up "
            "rolling windows (e.g. 260 ≈ 1 year of trading days). "
            "These dates are fetched but excluded from the output."
        ),
    )
    parser.add_argument(
        "--bbg-host", default="localhost",
        help="Bloomberg server host (default: localhost).",
    )
    parser.add_argument(
        "--bbg-port", type=int, default=8194,
        help="Bloomberg server port (default: 8194).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files. Overrides config.OUTPUT_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from fx_factor_analysis.config import (
        FACTOR_RETURNS_CSV,
        FACTOR_RETURNS_PARQUET,
        OUTPUT_DIR,
    )
    from fx_factor_analysis.construction.factor_returns import build_all_factor_returns
    from fx_factor_analysis.data.bbg_fetcher import BBGFetcher

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Extend fetch window backward for rolling window warm-up
    fetch_start = (
        pd.Timestamp(args.start) - timedelta(days=args.warmup)
    ).strftime("%Y-%m-%d")
    fetch_end = args.end

    logger.info("=" * 60)
    logger.info("FX Factor Construction")
    logger.info("  Analysis window : %s → %s", args.start, args.end)
    logger.info("  Fetch window    : %s → %s (incl. %d-day warmup)", fetch_start, fetch_end, args.warmup)
    logger.info("  Output dir      : %s", output_dir)
    logger.info("=" * 60)

    # --- 1. Pull Bloomberg data ---
    with BBGFetcher(host=args.bbg_host, port=args.bbg_port) as fetcher:
        data = fetcher.fetch_all(start_date=fetch_start, end_date=fetch_end)

    # --- 2. Build factor returns ---
    factor_returns: pd.DataFrame = build_all_factor_returns(data)

    # --- 3. Trim warm-up period from output ---
    analysis_start = pd.Timestamp(args.start)
    factor_returns = factor_returns[factor_returns.index >= analysis_start]

    logger.info(
        "Output: %d daily observations, %d factors, %d NaN cells.",
        len(factor_returns),
        factor_returns.shape[1],
        factor_returns.isna().sum().sum(),
    )

    # --- 4. Save ---
    csv_path = os.path.join(output_dir, FACTOR_RETURNS_CSV)
    parquet_path = os.path.join(output_dir, FACTOR_RETURNS_PARQUET)

    factor_returns.to_csv(csv_path)
    logger.info("Saved CSV     → %s", csv_path)

    try:
        factor_returns.to_parquet(parquet_path)
        logger.info("Saved Parquet → %s", parquet_path)
    except ImportError:
        logger.warning("pyarrow/fastparquet not installed — skipping Parquet output.")

    logger.info("Done.")
    return factor_returns


if __name__ == "__main__":
    main()
