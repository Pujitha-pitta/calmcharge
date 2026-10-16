"""Fit a small ridge-regression model without scikit-learn compiled extensions."""

from math import fsum, sqrt
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/se4_hourly.parquet")
PRICE_COLUMN = "price_eur_mwh"
TRAIN_FRACTION = 0.70
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "price_lag_1h",
    "price_lag_24h",
    "price_lag_168h",
    "price_mean_previous_24h",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features from calendar fields and prices observed before each hour."""
    index = frame.index
    features = pd.DataFrame(index=index)
    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["month"] = index.month
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)

    prices = frame[PRICE_COLUMN].astype(float)
    features["price_lag_1h"] = prices.shift(1)
    features["price_lag_24h"] = prices.shift(24)
    features["price_lag_168h"] = prices.shift(168)
    features["price_mean_previous_24h"] = prices.shift(1).rolling(24).mean()
    return features


def solve_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small square system with Gaussian elimination and pivoting."""
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Could not solve regression equations (singular matrix).")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        divisor = augmented[column][column]
        for j in range(column, size + 1):
            augmented[column][j] /= divisor

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for j in range(column, size + 1):
                augmented[row][j] -= factor * augmented[column][j]

    return [augmented[row][size] for row in range(size)]


def fit_ridge(train_x: list[list[float]], train_y: list[float], penalty: float = 1.0) -> list[float]:
    """Fit standardized ridge regression; the intercept is not penalized."""
    feature_count = len(train_x[0])
    means = [fsum(row[col] for row in train_x) / len(train_x) for col in range(feature_count)]
    scales = []
    for col in range(feature_count):
        variance = fsum((row[col] - means[col]) ** 2 for row in train_x) / len(train_x)
        scales.append(sqrt(variance) or 1.0)
    y_mean = fsum(train_y) / len(train_y)

    width = feature_count + 1
    gram = [[0.0] * width for _ in range(width)]
    target = [0.0] * width

    for row, y_value in zip(train_x, train_y):
        normalized = [1.0] + [
            (row[col] - means[col]) / scales[col] for col in range(feature_count)
        ]
        centered_y = y_value - y_mean
        for i in range(width):
            target[i] += normalized[i] * centered_y
            for j in range(width):
                gram[i][j] += normalized[i] * normalized[j]

    for i in range(1, width):
        gram[i][i] += penalty

    coefficients = solve_system(gram, target)
    # Save intercept, coefficients, feature means/scales, and target mean.
    return [y_mean + coefficients[0]] + coefficients[1:] + means + scales


def predict(model: list[float], rows: list[list[float]]) -> list[float]:
    feature_count = len(rows[0])
    intercept = model[0]
    coefficients = model[1 : 1 + feature_count]
    means = model[1 + feature_count : 1 + 2 * feature_count]
    scales = model[1 + 2 * feature_count : 1 + 3 * feature_count]
    predictions = []
    for row in rows:
        predictions.append(
            intercept
            + fsum(
                coefficients[col] * (row[col] - means[col]) / scales[col]
                for col in range(feature_count)
            )
        )
    return predictions


def score(actual: list[float], predicted: list[float]) -> tuple[float, float]:
    errors = [prediction - target for target, prediction in zip(actual, predicted)]
    mae = fsum(abs(error) for error in errors) / len(errors)
    rmse = sqrt(fsum(error * error for error in errors) / len(errors))
    return mae, rmse


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
    split_time = frame.index[int(len(frame) * TRAIN_FRACTION)]
    train = usable.loc[usable.index < split_time]
    test = usable.loc[usable.index >= split_time]
    if train.empty or test.empty:
        raise ValueError("Not enough data for a chronological train/test split.")

    train_x = train[FEATURE_COLUMNS].values.tolist()
    train_y = train["actual"].astype(float).tolist()
    test_x = test[FEATURE_COLUMNS].values.tolist()
    model = fit_ridge(train_x, train_y)
    model_prediction = predict(model, test_x)

    all_prices = frame[PRICE_COLUMN].astype(float)
    rolling_average = all_prices.groupby(frame.index.hour).transform(
        lambda hourly: hourly.shift(1).rolling(7, min_periods=7).mean()
    )
    method_predictions = {
        "Yesterday (same UTC hour)": all_prices.shift(24).reindex(test.index).tolist(),
        "Last week (same UTC hour)": all_prices.shift(168).reindex(test.index).tolist(),
        "Previous 7 days (same UTC hour average)": rolling_average.reindex(test.index).tolist(),
        "Ridge regression model": model_prediction,
    }

    actual_values = test["actual"].astype(float).tolist()
    results = []
    for name, values in method_predictions.items():
        pairs = [
            (target, prediction)
            for target, prediction in zip(actual_values, values)
            if pd.notna(prediction)
        ]
        targets = [pair[0] for pair in pairs]
        predictions = [float(pair[1]) for pair in pairs]
        mae, rmse = score(targets, predictions)
        results.append((name, len(pairs), mae, rmse))

    print("One-hour-ahead experiment: each prediction uses earlier observed prices.")
    print("Calendar fields and past price lags are used; future prices are not used.")
    print(f"Training period ends before: {split_time}")
    print(f"Evaluation period: {test.index[0]} to {test.index[-1]}")
    print("\nScores on the evaluation period (lower is better):")
    print(f"{'Method':48} {'Hours':>7} {'MAE':>12} {'RMSE':>12}")
    for name, hours, mae, rmse in sorted(results, key=lambda row: row[2]):
        print(f"{name:48} {hours:7,d} {mae:12.2f} {rmse:12.2f}")


if __name__ == "__main__":
    main()
