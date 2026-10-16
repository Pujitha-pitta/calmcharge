"""Train a leakage-aware one-hour-ahead price model and compare benchmarks."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path("data/processed/se4_hourly.parquet")
PRICE_COLUMN = "price_eur_mwh"
TRAIN_FRACTION = 0.70


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features from calendar fields and prices observed before each hour."""
    index = frame.index
    features = pd.DataFrame(index=index)
    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["month"] = index.month
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)

    # Only past prices are used. The rolling mean excludes the current hour.
    prices = frame[PRICE_COLUMN].astype(float)
    features["price_lag_1h"] = prices.shift(1)
    features["price_lag_24h"] = prices.shift(24)
    features["price_lag_168h"] = prices.shift(168)
    features["price_mean_previous_24h"] = prices.shift(1).rolling(24).mean()
    return features


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the Phase 1 ingestion command first."
        )

    frame = pd.read_parquet(DATA_PATH).sort_index()
    if PRICE_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing required column: {PRICE_COLUMN}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    frame = frame[[PRICE_COLUMN]].dropna()
    features = make_features(frame)
    model_data = features.join(frame[PRICE_COLUMN].rename("actual")).dropna()

    split_at = int(len(frame) * TRAIN_FRACTION)
    split_time = frame.index[split_at]
    train = model_data.loc[model_data.index < split_time]
    test = model_data.loc[model_data.index >= split_time]
    if train.empty or test.empty:
        raise ValueError("Not enough data for a chronological train/test split.")

    feature_columns = list(features.columns)
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[feature_columns], train["actual"])
    model_prediction = pd.Series(model.predict(test[feature_columns]), index=test.index)

    actual = test["actual"]
    baseline_predictions = pd.DataFrame(index=test.index)
    baseline_predictions["Yesterday (same UTC hour)"] = frame[PRICE_COLUMN].shift(24).reindex(test.index)
    baseline_predictions["Last week (same UTC hour)"] = frame[PRICE_COLUMN].shift(168).reindex(test.index)
    baseline_predictions["Previous 7 days (same UTC hour average)"] = frame[PRICE_COLUMN].groupby(
        frame.index.hour
    ).transform(lambda hourly: hourly.shift(1).rolling(7, min_periods=7).mean()).reindex(test.index)
    baseline_predictions["HistGradientBoosting model"] = model_prediction

    rows = []
    for name in baseline_predictions.columns:
        valid = pd.concat(
            [actual.rename("actual"), baseline_predictions[name].rename("predicted")],
            axis=1,
        ).dropna()
        rows.append(
            {
                "Method": name,
                "Evaluated hours": len(valid),
                "MAE (EUR/MWh)": mean_absolute_error(valid["actual"], valid["predicted"]),
                "RMSE (EUR/MWh)": np.sqrt(mean_squared_error(valid["actual"], valid["predicted"])),
            }
        )

    scores = pd.DataFrame(rows).sort_values("MAE (EUR/MWh)")
    print("One-hour-ahead experiment: each prediction uses earlier observed prices.")
    print("Calendar features and past price lags are used; future prices are not used.")
    print(f"Training period ends before: {split_time}")
    print(f"Evaluation period: {test.index[0]} to {test.index[-1]}")
    print("\nScores on the same evaluation period (lower is better):")
    print(scores.to_string(index=False, formatters={
        "MAE (EUR/MWh)": "{:.2f}".format,
        "RMSE (EUR/MWh)": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
