"""Bring the committed data snapshot up to date (run daily by GitHub Actions).

Downloads only the SE4 prices published since the snapshot, then forecasts the
newly completed days and appends them to the year-long backtest. Both files
only grow by a few rows, so each daily commit stays small.

Run: python -m src.refresh_snapshot
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from src.day_ahead import (
    PRICE_COLUMN,
    SNAPSHOT_BACKTEST,
    SNAPSHOT_PRICES,
    daily_grid,
    extend_backtest,
    load_prices,
)


def main() -> None:
    from src.ingest import fetch_se4_prices

    prices = load_prices(path=SNAPSHOT_PRICES.with_suffix(".parquet"), cache=SNAPSHOT_PRICES)
    now = pd.Timestamp.now(tz="UTC")
    recent = fetch_se4_prices(prices.index.max() - timedelta(days=2), now.normalize() + timedelta(days=2))
    merged = pd.concat([prices, recent[PRICE_COLUMN].astype(float)])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    added_hours = len(merged) - len(prices)
    merged.to_frame(PRICE_COLUMN).to_csv(SNAPSHOT_PRICES)

    predictions = pd.read_csv(
        SNAPSHOT_BACKTEST, index_col=["date", "hour"], parse_dates=["date"], float_precision="round_trip"
    )
    before = predictions.index.get_level_values("date").max()
    extended = extend_backtest(daily_grid(merged), predictions)
    after = extended.index.get_level_values("date").max()
    extended.to_csv(SNAPSHOT_BACKTEST)

    print(f"Prices: +{added_hours} hours, now up to {merged.index.max()}")
    print(f"Backtest: {before.date()} -> {after.date()} ({extended.index.get_level_values('date').nunique()} days)")


if __name__ == "__main__":
    main()
