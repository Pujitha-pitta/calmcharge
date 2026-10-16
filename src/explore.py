"""Create simple exploratory summaries and charts for the CalmCharge hourly dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/processed/se4_hourly.parquet")
OUTPUT_DIR = Path("reports/phase2")
PRICE_COLUMN = "price_eur_mwh"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the Phase 1 ingestion command first."
        )

    data = pd.read_parquet(DATA_PATH)
    if PRICE_COLUMN not in data.columns:
        raise ValueError(f"Dataset is missing required column: {PRICE_COLUMN}")

    prices = data[PRICE_COLUMN].dropna().copy()
    if prices.empty:
        raise ValueError("The dataset contains no price values to explore.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, utc=True)
    elif prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")

    # The project reports Swedish prices, so display calendar groupings in Sweden time.
    local_prices = prices.tz_convert("Europe/Stockholm")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rows in dataset: {len(data):,}")
    print(f"Price observations: {len(prices):,}")
    print(f"Time range (UTC): {prices.index.min()} to {prices.index.max()}")
    print(f"Mean price: {prices.mean():.2f} EUR/MWh")
    print(f"Median price: {prices.median():.2f} EUR/MWh")
    print(f"Minimum price: {prices.min():.2f} EUR/MWh")
    print(f"Maximum price: {prices.max():.2f} EUR/MWh")
    print(f"Negative-price hours: {(prices < 0).sum():,}")

    hourly = local_prices.groupby(local_prices.index.hour).mean()
    weekday = local_prices.groupby(local_prices.index.dayofweek).mean()
    monthly = local_prices.groupby(local_prices.index.month).mean()

    charts = [
        (
            hourly,
            [f"{hour:02d}:00" for hour in hourly.index],
            "Average electricity price by hour (Sweden time)",
            "Hour of day",
            "average_by_hour.png",
            "o",
        ),
        (
            weekday,
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Average electricity price by weekday (Sweden time)",
            "Day of week",
            "average_by_weekday.png",
            "o",
        ),
        (
            monthly,
            [pd.Timestamp(2000, month, 1).strftime("%b") for month in monthly.index],
            "Average electricity price by month",
            "Month",
            "average_by_month.png",
            "o",
        ),
    ]

    for values, labels, title, xlabel, filename, marker in charts:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(labels, values.to_numpy(), marker=marker, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average price (EUR/MWh)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        output_path = OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"Saved chart: {output_path}")

    print(f"\nCharts are in {OUTPUT_DIR}. These are descriptive averages, not forecasts.")


if __name__ == "__main__":
    main()
