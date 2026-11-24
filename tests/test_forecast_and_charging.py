"""Tests for the charging optimiser, the day-ahead forecast, and the savings backtest."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest_savings import Scenario, simulate, summarise, window_costs
from src.charging import ChargeRequest, add_household_cost, charging_window, energy_from_soc, plan
from src.day_ahead import (
    LAG_FEATURES,
    build_features,
    daily_grid,
    extend_backtest,
    forecast_next_day,
    walk_forward,
)

TZ = "Europe/Stockholm"


# ------------------------------------------------------------------ helpers


def synthetic_prices(days: int = 200, start: str = "2025-01-01", seed: int = 0) -> pd.Series:
    """Hourly UTC prices with a daily shape (cheap nights), weekly pattern and noise."""
    index = pd.date_range(pd.Timestamp(start, tz=TZ), periods=days * 24, freq="h").tz_convert("UTC")
    local = index.tz_convert(TZ)
    rng = np.random.default_rng(seed)
    daily_shape = 60 + 40 * np.sin((local.hour - 6) / 24 * 2 * np.pi)
    weekend = np.where(local.dayofweek >= 5, -15, 0)
    level = 20 * np.sin(np.arange(len(index)) / (24 * 30))
    return pd.Series(daily_shape + weekend + level + rng.normal(0, 5, len(index)), index=index, name="price_eur_mwh")


def local_hours(start: str, hours: int, values) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start, tz=TZ), periods=hours, freq="h")
    return pd.Series(values, index=index, dtype=float)


# ------------------------------------------------------------------ charging


def test_cheapest_plan_picks_lowest_hours_and_partial_last_hour():
    prices = local_hours("2026-01-10 18:00", 6, [90, 80, 10, 20, 30, 70])
    request = ChargeRequest(energy_kwh=10, charger_kw=4, efficiency=1.0)  # 2.5 hours
    schedule = plan(prices, request)
    assert list(schedule["price_eur_mwh"]) == [10, 20, 30]
    assert schedule["battery_kwh"].tolist() == [4, 4, 2]
    assert schedule["cost_eur"].sum() == pytest.approx((10 * 4 + 20 * 4 + 30 * 2) / 1000)


def test_immediate_plan_charges_from_plug_in():
    prices = local_hours("2026-01-10 18:00", 6, [90, 80, 10, 20, 30, 70])
    schedule = plan(prices, ChargeRequest(8, 4, 1.0), strategy="immediate")
    assert list(schedule["price_eur_mwh"]) == [90, 80]


def test_efficiency_increases_grid_energy():
    prices = local_hours("2026-01-10 00:00", 4, [10, 10, 10, 10])
    schedule = plan(prices, ChargeRequest(9, 10, 0.9))
    assert schedule["grid_kwh"].sum() == pytest.approx(10)


def test_window_crosses_midnight():
    prices = local_hours("2026-01-10 00:00", 48, range(48))
    window = charging_window(
        pd.Timestamp("2026-01-10 18:00", tz=TZ), pd.Timestamp("2026-01-11 07:00", tz=TZ), prices
    )
    assert len(window) == 13
    assert window.index[0].hour == 18 and window.index[-1].hour == 6


def test_forecast_ranking_chooses_hours_but_actual_prices_set_cost():
    actual = local_hours("2026-01-10 22:00", 3, [10, 50, 100])
    forecast = local_hours("2026-01-10 22:00", 3, [100, 50, 10])  # a wrong forecast
    schedule = plan(actual, ChargeRequest(1, 1, 1.0), ranking_prices=forecast)
    assert schedule["price_eur_mwh"].iloc[0] == 100  # pays the real price of the bad choice


def test_rejects_window_too_short():
    prices = local_hours("2026-01-10 00:00", 2, [1, 2])
    with pytest.raises(ValueError, match="can add at most"):
        plan(prices, ChargeRequest(energy_kwh=50, charger_kw=7.2))


def test_energy_from_soc_and_validation():
    assert energy_from_soc(60, 30, 80) == pytest.approx(30)
    with pytest.raises(ValueError):
        energy_from_soc(60, 80, 30)
    with pytest.raises(ValueError):
        ChargeRequest(energy_kwh=10, efficiency=1.5)


def test_household_cost_adds_fees_and_vat():
    schedule = plan(local_hours("2026-01-10 00:00", 1, [100]), ChargeRequest(1, 1, 1.0))
    priced = add_household_cost(schedule, sek_per_eur=10, grid_fee_sek_kwh=0.5, vat_percent=25)
    # 1 kWh * (100 EUR/MWh * 10 SEK/EUR / 1000 + 0.5 SEK) * 1.25
    assert priced["total_sek"].sum() == pytest.approx(1.875)


# ------------------------------------------------------------------ forecast


def test_daily_grid_normalises_daylight_saving_days_to_24_hours():
    prices = synthetic_prices(days=10, start="2026-03-25")  # includes spring-forward 2026-03-29
    grid = daily_grid(prices)
    assert list(grid.columns) == list(range(24))
    assert grid.loc["2026-03-29"].notna().all()


def test_features_only_use_information_published_before_the_forecast_day():
    prices = synthetic_prices(days=30)
    grid = daily_grid(prices)
    features = build_features(grid)
    target = pd.Timestamp("2025-01-20")
    row = features.loc[(target, 17)]
    assert row["lag_1d"] == pytest.approx(grid.loc[target - timedelta(days=1), 17])
    assert row["lag_7d"] == pytest.approx(grid.loc[target - timedelta(days=7), 17])
    assert row["d1_max"] == pytest.approx(grid.loc[target - timedelta(days=1)].max())

    # Changing the target day's own prices must not change any of its features.
    tampered = grid.copy()
    tampered.loc[target] = 9999.0
    changed = build_features(tampered).loc[target, LAG_FEATURES]
    pd.testing.assert_frame_equal(changed, features.loc[target, LAG_FEATURES])


def test_walk_forward_never_trains_on_the_day_it_forecasts():
    grid = daily_grid(synthetic_prices(days=200))
    eval_start = grid.index.max() - timedelta(days=13)
    clean = walk_forward(grid, eval_start, model_keys=["ridge"], min_train_days=60)

    # Corrupt the last day's actual prices: its forecast must not change.
    corrupted_grid = grid.copy()
    corrupted_grid.loc[grid.index.max()] += 500
    corrupted = walk_forward(corrupted_grid, eval_start, model_keys=["ridge"], min_train_days=60)
    last = grid.index.max()
    np.testing.assert_allclose(
        clean.predictions.loc[last, "pred_Ridge regression"],
        corrupted.predictions.loc[last, "pred_Ridge regression"],
    )


def test_model_beats_naive_baseline_on_patterned_prices():
    grid = daily_grid(synthetic_prices(days=200))
    result = walk_forward(grid, grid.index.max() - timedelta(days=29), model_keys=["ridge"], min_train_days=60)
    metrics = result.metrics().set_index("method")
    assert metrics.loc["Ridge regression", "mae"] < metrics.loc["Same hour, previous day", "mae"]


def test_forecast_next_day_covers_every_clock_hour_with_ordered_ranges():
    prices = synthetic_prices(days=200)
    grid = daily_grid(prices)
    backtest = walk_forward(grid, grid.index.max() - timedelta(days=39), model_keys=["ridge"], min_train_days=60)
    forecast = forecast_next_day(prices, backtest, model_key="ridge")
    assert len(forecast) == 24
    assert forecast.index[0].date() == (grid.index.max() + timedelta(days=1)).date()
    assert (forecast["low"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["high"]).all()


# ------------------------------------------------------------------ savings


def test_extend_backtest_appends_new_days_without_touching_old_rows():
    grid = daily_grid(synthetic_prices(days=200))
    last = grid.index.max()
    old = walk_forward(grid.loc[: last - timedelta(days=3)], last - timedelta(days=32),
                       model_keys=["ridge"], min_train_days=60).predictions
    extended = extend_backtest(grid, old, keep_days=30)

    new_days = sorted(set(extended.index.get_level_values("date")) - set(old.index.get_level_values("date")))
    assert new_days == [last - timedelta(days=2), last - timedelta(days=1), last]
    assert extended.index.get_level_values("date").nunique() == 30
    kept = extended.index.intersection(old.index)
    pd.testing.assert_frame_equal(extended.loc[kept], old.loc[kept])
    assert extended.loc[last, ["forecast", "low", "high"]].notna().all().all()

    # The new days' forecasts must not depend on their own actual prices.
    tampered = grid.copy()
    tampered.loc[last] += 500
    np.testing.assert_allclose(
        extend_backtest(tampered, old, keep_days=30).loc[last, "forecast"],
        extended.loc[last, "forecast"],
    )


def test_extend_backtest_is_a_no_op_when_nothing_is_new():
    grid = daily_grid(synthetic_prices(days=200))
    old = walk_forward(grid, grid.index.max() - timedelta(days=9), model_keys=["ridge"], min_train_days=60).predictions
    pd.testing.assert_frame_equal(extend_backtest(grid, old), old)


@pytest.mark.parametrize("energy", [3.0, 10.0, 22.5, 40.0])
def test_vectorised_costs_match_the_planner_exactly(energy):
    rng = np.random.default_rng(1)
    actual = rng.normal(80, 40, size=(25, 13)).round()  # rounding creates price ties
    ranking = actual + rng.normal(0, 20, size=actual.shape)
    request = ChargeRequest(energy_kwh=energy, charger_kw=7.2, efficiency=0.9)
    fast = window_costs(actual, ranking, request)
    for night in range(actual.shape[0]):
        window = local_hours("2026-01-10 18:00", 13, actual[night])
        rank = local_hours("2026-01-10 18:00", 13, ranking[night])
        assert fast[night] == pytest.approx(plan(window, request, ranking_prices=rank)["cost_eur"].sum())
        assert window_costs(actual[[night]], actual[[night]], request)[0] == pytest.approx(
            plan(window, request)["cost_eur"].sum()
        )


def test_savings_backtest_orders_strategies_sensibly():
    grid = daily_grid(synthetic_prices(days=200))
    backtest = walk_forward(grid, grid.index.max() - timedelta(days=29), model_keys=["ridge"], min_train_days=60)
    scenario = Scenario(request=ChargeRequest(energy_kwh=15))
    nights = simulate(backtest.predictions, scenario)
    summary = summarise(nights, scenario)
    totals = summary["totals_eur"]
    assert totals["smart_published"] <= totals["smart_forecast"] <= totals["immediate"]
    assert summary["savings_pct"]["smart_published"] > 0
    assert len(nights) == 29  # the last evening has no next morning in the backtest
