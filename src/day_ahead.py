"""True day-ahead forecast of SE4 hourly prices.

The forecast for delivery day T is issued at 11:00 Swedish time on day T-1,
just before the noon day-ahead auction closes. At that moment the prices for
T-1 itself are already public (they were set the day before), but nothing about
day T is. Every feature below respects that cutoff:

* prices for the same hour on T-1, T-2 and T-7, and the 7-day same-hour mean
* level and spread of the full T-1 price curve
* calendar facts about day T (hour, weekday, Swedish holidays, season)

Validation is walk-forward: each forecast only uses a model trained on days
strictly before the forecast date, retrained every few days. Prediction ranges
come from the model's own recent out-of-sample errors, so their coverage is
measured honestly too.

Run: python -m src.day_ahead            (uses data/processed/se4_hourly.parquet)
     python -m src.day_ahead --fetch    (downloads prices instead)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "Europe/Stockholm"
DATA_PATH = Path("data/processed/se4_hourly.parquet")
REPORT_DIR = Path("reports")
PRICE_COLUMN = "price_eur_mwh"
INTERVAL_QUANTILES = (0.10, 0.90)

LAG_FEATURES = [
    "lag_1d", "lag_2d", "lag_7d", "mean_7d",
    "d1_mean", "d1_min", "d1_max", "d1_std", "d1_last",
]
CALENDAR_FEATURES = ["is_offday", "month_sin", "month_cos"]


# --------------------------------------------------------------------------- data


PRICE_CACHE = Path("data/processed/se4_prices.csv")


def load_prices(path: Path = DATA_PATH, cache: Path = PRICE_CACHE) -> pd.Series:
    """Read hourly prices from the ingestion parquet, or from the CSV price cache.

    The CSV cache exists because some Windows Application Control policies block
    pyarrow's Parquet library; a plain CSV always works.
    """
    frame = None
    if path.exists():
        try:
            frame = pd.read_parquet(path)
        except ImportError:
            frame = None
    if frame is None and cache.exists():
        frame = pd.read_csv(cache, index_col=0)
    if frame is None:
        raise FileNotFoundError(
            f"Could not find {path} or {cache}. Run `python -m src.day_ahead --fetch` first."
        )
    frame = frame.sort_index()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    frame.index = index
    return frame[PRICE_COLUMN].astype(float)


def fetch_prices(days: int, cache: Path = PRICE_CACHE) -> pd.Series:
    """Download the last ``days`` days of SE4 prices (plus tomorrow, if published) and cache them."""
    from src.ingest import fetch_se4_prices

    end = pd.Timestamp.now(tz="UTC").normalize() + timedelta(days=2)
    prices = fetch_se4_prices(end - timedelta(days=days), end)[PRICE_COLUMN].astype(float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_frame(PRICE_COLUMN).to_csv(cache)
    return prices


def daily_grid(prices: pd.Series) -> pd.DataFrame:
    """Reshape hourly UTC prices into one row per Swedish date and 24 hour columns.

    Daylight-saving days are normalised to 24 local hours: the missing spring
    hour is interpolated and the doubled autumn hour is averaged. Days with
    more than one missing hour are left empty rather than guessed.
    """
    local = prices.dropna().tz_convert(LOCAL_TIMEZONE)
    frame = pd.DataFrame(
        {
            "date": local.index.normalize().tz_localize(None),
            "hour": local.index.hour,
            "price": local.to_numpy(dtype=float),
        }
    )
    grid = frame.pivot_table(index="date", columns="hour", values="price", aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    grid = grid.loc[grid.notna().sum(axis=1) >= 23]
    grid = grid.interpolate(axis=1, limit=1, limit_direction="both")
    grid.columns.name = "hour"
    if grid.empty:
        return grid
    return grid.asfreq("D")


# ----------------------------------------------------------------------- features


def _stack(frame: pd.DataFrame, name: str) -> pd.Series:
    series = frame.stack(future_stack=True)
    series.name = name
    return series


def build_features(grid: pd.DataFrame, last_target: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per (target date, hour) using only information public at 11:00 on T-1."""
    end = grid.index.max() if last_target is None else max(grid.index.max(), last_target)
    full = grid.reindex(pd.date_range(grid.index.min(), end, freq="D"))

    known = full.shift(1)  # row T holds the curve for T-1, the latest public prices
    parts = [
        _stack(full, "actual"),
        _stack(known, "lag_1d"),
        _stack(full.shift(2), "lag_2d"),
        _stack(full.shift(7), "lag_7d"),
        _stack(known.rolling(7, min_periods=5).mean(), "mean_7d"),
    ]
    features = pd.concat(parts, axis=1)
    features.index.names = ["date", "hour"]

    stats = pd.DataFrame(
        {
            "d1_mean": known.mean(axis=1),
            "d1_min": known.min(axis=1),
            "d1_max": known.max(axis=1),
            "d1_std": known.std(axis=1),
            "d1_last": known[23],
        }
    )
    features = features.join(stats, on="date")

    dates = features.index.get_level_values("date")
    hours = features.index.get_level_values("hour")
    swedish = holidays.country_holidays("SE", years=range(dates.min().year, dates.max().year + 1))
    is_holiday = np.array([day.date() in swedish for day in dates])
    features["hour"] = hours
    features["weekday"] = dates.dayofweek
    features["is_offday"] = ((dates.dayofweek >= 5) | is_holiday).astype(float)
    features["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    return features


def _design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Linear-model inputs: lags, calendar, and hour shapes for work days and off days."""
    hour_dummies = pd.get_dummies(features["hour"], prefix="h", dtype=float)
    hour_dummies = hour_dummies.reindex(columns=[f"h_{h}" for h in range(24)], fill_value=0.0)
    offday_hours = hour_dummies.mul(features["is_offday"], axis=0).add_prefix("off_")
    weekday = pd.get_dummies(features["weekday"], prefix="wd", dtype=float)
    weekday = weekday.reindex(columns=[f"wd_{d}" for d in range(7)], fill_value=0.0)
    # Let the effect of yesterday's price depend on the hour of day.
    lag_by_hour = hour_dummies.mul(features["lag_1d"], axis=0).add_prefix("lag1x_")
    return pd.concat(
        [features[LAG_FEATURES + CALENDAR_FEATURES], hour_dummies, offday_hours, weekday, lag_by_hour],
        axis=1,
    ).astype(float)


# ------------------------------------------------------------------------- models


class RidgeForecaster:
    """Closed-form ridge regression on standardised inputs (NumPy only)."""

    name = "Ridge regression"

    def __init__(self, penalty: float = 3.0) -> None:
        self.penalty = penalty

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeForecaster":
        x = _design_matrix(features)
        self.columns = x.columns
        self.mean = x.mean()
        self.scale = x.std().replace(0, 1.0).fillna(1.0)
        z = ((x - self.mean) / self.scale).to_numpy()
        y = target.to_numpy(dtype=float)
        self.intercept = y.mean()
        gram = z.T @ z + self.penalty * np.eye(z.shape[1])
        self.coef = np.linalg.solve(gram, z.T @ (y - self.intercept))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        x = _design_matrix(features).reindex(columns=self.columns, fill_value=0.0)
        z = ((x - self.mean) / self.scale).to_numpy()
        return self.intercept + z @ self.coef


class LightGBMForecaster:
    """Gradient-boosted trees; captures non-linear effects such as evening spikes."""

    name = "LightGBM"
    columns = LAG_FEATURES + CALENDAR_FEATURES + ["hour", "weekday"]

    params = {
        "objective": "l1",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "seed": 0,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    rounds = 500

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMForecaster":
        # The native API avoids LightGBM's scikit-learn wrapper (and that dependency).
        import lightgbm as lgb

        data = lgb.Dataset(features[self.columns].astype(float), label=target.astype(float))
        self.booster = lgb.train(self.params, data, num_boost_round=self.rounds)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(features[self.columns].astype(float))


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:  # missing or blocked compiled library
        return False
    return True


def make_models() -> dict[str, type]:
    models: dict[str, type] = {"ridge": RidgeForecaster}
    if lightgbm_available():
        models["lightgbm"] = LightGBMForecaster
    return models


BASELINES = {
    "Same hour, previous day": "lag_1d",
    "Same hour, one week earlier": "lag_7d",
    "Same hour, 7-day average": "mean_7d",
}


# --------------------------------------------------------------------- evaluation


@dataclass
class Backtest:
    predictions: pd.DataFrame  # index (date, hour); actual, baselines, model columns, ranges
    model_key: str

    def metrics(self) -> pd.DataFrame:
        frame = self.predictions
        rows = []
        for column in [c for c in frame.columns if c.startswith("pred_")]:
            error = frame[column] - frame["actual"]
            rows.append(
                {
                    "method": column.removeprefix("pred_"),
                    "mae": error.abs().mean(),
                    "rmse": float(np.sqrt((error**2).mean())),
                }
            )
        return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)

    def interval_coverage(self) -> float:
        frame = self.predictions.dropna(subset=["low", "high"])
        inside = (frame["actual"] >= frame["low"]) & (frame["actual"] <= frame["high"])
        return float(inside.mean())


def walk_forward(
    grid: pd.DataFrame,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp | None = None,
    *,
    model_keys: list[str] | None = None,
    retrain_every_days: int = 7,
    min_train_days: int = 120,
    interval_window_days: int = 90,
) -> Backtest:
    """Forecast each day in [eval_start, eval_end) with models trained only on earlier days."""
    features = build_features(grid)
    usable = features.dropna(subset=["actual"] + LAG_FEATURES)
    dates = usable.index.get_level_values("date")
    eval_end = dates.max() + timedelta(days=1) if eval_end is None else eval_end
    test_dates = pd.DatetimeIndex(sorted(set(dates[(dates >= eval_start) & (dates < eval_end)])))
    if test_dates.empty:
        raise ValueError("No complete days to evaluate in the requested period.")
    if dates.min() > test_dates[0] - timedelta(days=min_train_days):
        raise ValueError(f"Need at least {min_train_days} days of history before {test_dates[0].date()}.")

    available = make_models()
    keys = list(available) if model_keys is None else model_keys
    test = usable.loc[dates.isin(test_dates)].copy()
    for label, column in BASELINES.items():
        test[f"pred_{label}"] = test[column]

    for key in keys:
        cls = available[key]
        predictions = pd.Series(np.nan, index=test.index)
        for i in range(0, len(test_dates), retrain_every_days):
            chunk = test_dates[i : i + retrain_every_days]
            train = usable.loc[dates < chunk[0]]  # strictly before the first forecast day
            model = cls().fit(train, train["actual"])
            rows = test.index.get_level_values("date").isin(chunk)
            predictions[rows] = model.predict(test.loc[rows])
        test[f"pred_{cls.name}"] = predictions

    best = min(keys, key=lambda k: (test[f"pred_{available[k].name}"] - test["actual"]).abs().mean())
    best_column = f"pred_{available[best].name}"
    test["forecast"] = test[best_column]

    # Ranges: per-hour quantiles of errors from earlier days only (shifted one day).
    residual = (test["actual"] - test["forecast"]).unstack("hour")
    residual = residual.reindex(pd.date_range(residual.index.min(), residual.index.max(), freq="D"))
    window = residual.rolling(interval_window_days, min_periods=30)
    low_q = _stack(window.quantile(INTERVAL_QUANTILES[0]).shift(1), "low_q")
    high_q = _stack(window.quantile(INTERVAL_QUANTILES[1]).shift(1), "high_q")
    low_q.index.names = high_q.index.names = ["date", "hour"]
    test["low"] = test["forecast"] + low_q.reindex(test.index).to_numpy()
    test["high"] = test["forecast"] + high_q.reindex(test.index).to_numpy()

    keep = ["actual", "forecast", "low", "high"] + [c for c in test.columns if c.startswith("pred_")]
    return Backtest(predictions=test[keep], model_key=best)


# ----------------------------------------------------------------------- forecast


def forecast_next_day(
    prices: pd.Series,
    backtest: Backtest | None = None,
    model_key: str | None = None,
) -> pd.DataFrame:
    """Forecast the first Swedish day after the last fully published one.

    Returns 24 rows indexed by local timestamps with forecast, low and high.
    Ranges reuse the most recent out-of-sample errors from ``backtest``.
    """
    grid = daily_grid(prices)
    complete = grid.dropna()
    target = complete.index.max() + timedelta(days=1)
    features = build_features(grid.loc[: complete.index.max()], last_target=target)
    train = features.dropna(subset=["actual"] + LAG_FEATURES)
    query = features.xs(target, level="date", drop_level=False)
    if query[LAG_FEATURES].isna().any().any():
        raise ValueError("Not enough recent published prices to build a forecast.")

    available = make_models()
    key = model_key or (backtest.model_key if backtest else ("lightgbm" if "lightgbm" in available else "ridge"))
    if key not in available:
        key = "ridge"  # e.g. LightGBM can't load on this machine
    model = available[key]().fit(train, train["actual"])
    forecast = pd.Series(model.predict(query), index=range(24))

    low_q = pd.Series(0.0, index=range(24))
    high_q = pd.Series(0.0, index=range(24))
    if backtest is not None:
        recent = backtest.predictions
        recent_dates = recent.index.get_level_values("date")
        recent = recent.loc[recent_dates >= recent_dates.max() - timedelta(days=89)]
        residual = (recent["actual"] - recent["forecast"]).groupby(level="hour")
        low_q = residual.quantile(INTERVAL_QUANTILES[0])
        high_q = residual.quantile(INTERVAL_QUANTILES[1])

    # Real clock hours of the target day (23 or 25 of them on daylight-saving days).
    midnight = pd.Timestamp(target).tz_localize(LOCAL_TIMEZONE)
    next_midnight = (pd.Timestamp(target) + timedelta(days=1)).tz_localize(LOCAL_TIMEZONE)
    stamps = pd.date_range(midnight, next_midnight, freq="h", inclusive="left", name="time")
    hours = stamps.hour
    result = pd.DataFrame(
        {
            "forecast": forecast.reindex(hours).to_numpy(),
            "low": np.minimum(forecast + low_q, forecast).reindex(hours).to_numpy(),
            "high": np.maximum(forecast + high_q, forecast).reindex(hours).to_numpy(),
        },
        index=stamps,
    )
    result.attrs["model"] = key
    return result


# ---------------------------------------------------------------------------- CLI


def _save_chart(backtest: Backtest, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = backtest.predictions.reset_index()
    frame["time"] = pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"], unit="h")
    last = frame.loc[frame["time"] >= frame["time"].max() - timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(last["time"], last["low"], last["high"], color="#9ec5fe", alpha=0.5,
                    label="80% forecast range")
    ax.plot(last["time"], last["actual"], color="#111827", lw=1.4, label="Actual day-ahead price")
    ax.plot(last["time"], last["forecast"], color="#2563eb", lw=1.4, label="Forecast (issued day before)")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("SE4 day-ahead forecast vs. actual, last 14 days of the backtest", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward test of the SE4 day-ahead forecast.")
    parser.add_argument("--fetch", action="store_true", help="Download prices instead of reading parquet")
    parser.add_argument("--eval-days", type=int, default=365, help="Days to evaluate (default 365)")
    args = parser.parse_args()

    if args.fetch:
        prices = fetch_prices(args.eval_days + 400)
        print(f"Downloaded {len(prices):,} hourly prices and cached them in {PRICE_CACHE}")
    else:
        prices = load_prices()

    grid = daily_grid(prices)
    eval_start = grid.dropna().index.max() - timedelta(days=args.eval_days - 1)
    backtest = walk_forward(grid, eval_start)
    metrics = backtest.metrics()
    coverage = backtest.interval_coverage()
    days = backtest.predictions.index.get_level_values("date")

    print("Day-ahead forecast, walk-forward evaluation")
    print(f"Forecast issued 11:00 the day before; {days.nunique()} days, {days.min().date()} to {days.max().date()}")
    print(f"\n{'Method':34} {'MAE':>8} {'RMSE':>8}")
    for row in metrics.itertuples():
        print(f"{row.method:34} {row.mae:8.2f} {row.rmse:8.2f}")
    print(f"\n80% range coverage: {coverage:.1%} of actual prices fell inside the range")

    REPORT_DIR.mkdir(exist_ok=True)
    backtest.predictions.to_csv(REPORT_DIR / "day_ahead_backtest.csv")
    summary = {
        "first_day": str(days.min().date()),
        "last_day": str(days.max().date()),
        "days": int(days.nunique()),
        "best_model": backtest.model_key,
        "interval_coverage": coverage,
        "metrics": metrics.to_dict(orient="records"),
    }
    (REPORT_DIR / "day_ahead_metrics.json").write_text(json.dumps(summary, indent=2))
    _save_chart(backtest, REPORT_DIR / "day_ahead_forecast.png")
    print(f"Saved results and chart under {REPORT_DIR}/")


if __name__ == "__main__":
    main()
