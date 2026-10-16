"""Run three expanding-window validation folds for CalmCharge baselines and ridge model."""

from pathlib import Path

import pandas as pd

from src.train_model import (
    DATA_PATH,
    FEATURE_COLUMNS,
    PRICE_COLUMN,
    TRAIN_FRACTION,
    fit_ridge,
    make_features,
    predict,
    score,
)


FOLD_COUNT = 3


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

    features = make_features(frame)[FEATURE_COLUMNS]
    usable = features.join(frame[PRICE_COLUMN].rename("actual")).dropna()
    n_rows = len(usable)
    first_test_start = int(n_rows * TRAIN_FRACTION)
    fold_width = (n_rows - first_test_start) // FOLD_COUNT
    if fold_width < 1:
        raise ValueError("Not enough rows for three chronological validation folds.")

    prices = frame[PRICE_COLUMN].astype(float)
    weekly_average = prices.groupby(frame.index.hour).transform(
        lambda hourly: hourly.shift(1).rolling(7, min_periods=7).mean()
    )

    methods = [
        "Ridge regression model",
        "Yesterday (same UTC hour)",
        "Previous 7 days (same UTC hour average)",
        "Last week (same UTC hour)",
    ]
    actual_by_method = {name: [] for name in methods}
    predicted_by_method = {name: [] for name in methods}

    print("Expanding-window evaluation (each fold trains only on earlier timestamps):")
    for fold in range(FOLD_COUNT):
        start = first_test_start + fold * fold_width
        end = n_rows if fold == FOLD_COUNT - 1 else first_test_start + (fold + 1) * fold_width
        train = usable.iloc[:start]
        test = usable.iloc[start:end]
        if train.empty or test.empty:
            raise ValueError(f"Fold {fold + 1} has no training or evaluation rows.")

        model = fit_ridge(
            train[FEATURE_COLUMNS].values.tolist(),
            train["actual"].astype(float).tolist(),
        )
        model_predictions = predict(model, test[FEATURE_COLUMNS].values.tolist())
        fold_prices = prices.reindex(test.index)
        fold_predictions = {
            "Ridge regression model": model_predictions,
            "Yesterday (same UTC hour)": prices.shift(24).reindex(test.index).tolist(),
            "Previous 7 days (same UTC hour average)": weekly_average.reindex(test.index).tolist(),
            "Last week (same UTC hour)": prices.shift(168).reindex(test.index).tolist(),
        }

        print(
            f"Fold {fold + 1}: train through {train.index[-1]} | "
            f"evaluate {test.index[0]} to {test.index[-1]} ({len(test):,} hours)"
        )
        for name in methods:
            predictions = fold_predictions[name]
            valid = [
                (float(actual), float(prediction))
                for actual, prediction in zip(fold_prices.tolist(), predictions)
                if pd.notna(actual) and pd.notna(prediction)
            ]
            actual_values = [pair[0] for pair in valid]
            predicted_values = [pair[1] for pair in valid]
            fold_mae, fold_rmse = score(actual_values, predicted_values)
            print(f"  {name}: MAE {fold_mae:.2f}, RMSE {fold_rmse:.2f}")
            actual_by_method[name].extend(actual_values)
            predicted_by_method[name].extend(predicted_values)

    print("\nCombined scores across all validation folds (lower is better):")
    print(f"{'Method':48} {'Hours':>7} {'MAE':>12} {'RMSE':>12}")
    combined = []
    for name in methods:
        mae, rmse = score(actual_by_method[name], predicted_by_method[name])
        combined.append((name, len(actual_by_method[name]), mae, rmse))
    for name, hours, mae, rmse in sorted(combined, key=lambda row: row[2]):
        print(f"{name:48} {hours:7,d} {mae:12.2f} {rmse:12.2f}")


if __name__ == "__main__":
    main()
