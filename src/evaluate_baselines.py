"""Evaluate three simple, time-ordered electricity-price forecasting baselines."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/se4_hourly.parquet")
PRICE_COLUMN = "price_eur_mwh"
TRAIN_FRACTION = 0.70


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the Phase 1 ingestion command first."
        )

    frame = pd.read_parquet(DATA_PATH)
    if PRICE_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing required column: {PRICE_COLUMN}")

    actual = frame[PRICE_COLUMN].dropna().sort_index().astype(float)
    if not isinstance(actual.index, pd.DatetimeIndex):
        actual.index = pd.to_datetime(actual.index, utc=True)
    elif actual.index.tz is None:
        actual.index = actual.index.tz_localize("UTC")

    if len(actual) < 24 * 14:
        raise ValueError("At least 14 days of hourly prices are needed.")

    # These simple baselines use only observations available before each prediction.
    predictions = pd.DataFrame(index=actual.index)
    predictions["Yesterday (same UTC hour)"] = actual.shift(24)
    predictions["Last week (same UTC hour)"] = actual.shift(24 * 7)
    predictions["Previous 7 days (same UTC hour average)"] = actual.groupby(
        actual.index.hour
    ).transform(lambda hourly: hourly.shift(1).rolling(7, min_periods=7).mean())

    split_at = int(len(actual) * TRAIN_FRACTION)
    test_index = actual.index[split_at:]
    results = []

    for name in predictions.columns:
        comparison = pd.concat(
            [actual.loc[test_index].rename("actual"), predictions.loc[test_index, name].rename("predicted")],
            axis=1,
        ).dropna()
        errors = comparison["predicted"] - comparison["actual"]
        results.append(
            {
                "Baseline": name,
                "Evaluated hours": len(comparison),
                "MAE (EUR/MWh)": float(errors.abs().mean()),
                "RMSE (EUR/MWh)": float(np.sqrt((errors**2).mean())),
            }
        )

    scores = pd.DataFrame(results).sort_values("MAE (EUR/MWh)")
    print("Chronological holdout: earliest 70% for history, latest 30% for evaluation")
    print(f"Evaluation period starts: {test_index[0]}")
    print(f"Evaluation period ends:   {test_index[-1]}")
    print("All forecasts use only earlier prices; no future values are used.")
    print("\nBaseline scores (lower MAE/RMSE is better):")
    print(scores.to_string(index=False, formatters={
        "MAE (EUR/MWh)": "{:.2f}".format,
        "RMSE (EUR/MWh)": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
