"""End-to-end checks of the web API on synthetic prices (no network)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import web
from src.day_ahead import daily_grid, forecast_next_day, load_forecast, save_forecast
from tests.test_forecast_and_charging import synthetic_prices


@pytest.fixture()
def client(monkeypatch, tmp_path):
    prices = synthetic_prices(days=200, start="2026-01-01")
    monkeypatch.setattr(web, "_load_and_top_up_prices", lambda: prices)
    monkeypatch.setattr(web, "BACKTEST_CACHE", tmp_path / "backtest.csv")
    monkeypatch.setattr(web, "BACKTEST_DAYS", 30)
    monkeypatch.setattr(web, "store", web.Store())
    monkeypatch.setattr(web, "load_forecast", lambda: None)  # ignore the committed snapshot
    return TestClient(web.app), prices


def test_plan_uses_forecast_for_unpublished_hours(client):
    api, prices = client
    last_day = daily_grid(prices).dropna().index.max().date()  # tomorrow is "unpublished"
    response = api.get("/api/plan", params={"plug_date": last_day.isoformat(), "plug_hour": 18, "depart_hour": 7})
    assert response.status_code == 200
    body = response.json()
    assert len(body["hours"]) == 13
    published_until = prices.index.max()
    expected = sum(pd.Timestamp(h["time"]) > published_until for h in body["hours"])
    assert body["forecast_hours"] == expected > 0
    assert all((h["source"] == "forecast") == (pd.Timestamp(h["time"]) > published_until) for h in body["hours"])
    assert sum(h["charge"] for h in body["hours"]) == body["hours_needed"]
    assert body["smart_cost"] <= body["naive_cost"]


def test_plan_rejects_dates_too_far_ahead(client):
    api, prices = client
    future = (prices.index.max() + pd.Timedelta(days=5)).date().isoformat()
    response = api.get("/api/plan", params={"plug_date": future})
    assert response.status_code == 422


def test_savings_and_forecast_endpoints(client):
    api, _ = client
    savings = api.get("/api/savings", params={"energy_kwh": 20}).json()
    assert savings["totals_eur"]["smart_published"] <= savings["totals_eur"]["immediate"]
    assert len(savings["series"]["nights"]) == len(savings["series"]["immediate"])

    forecast = api.get("/api/forecast").json()
    assert len(forecast["hours"]) == 24
    assert forecast["metrics"][0]["mae"] <= forecast["metrics"][-1]["mae"]


def test_homepage_is_served(client):
    api, _ = client
    page = api.get("/")
    assert page.status_code == 200
    assert "CalmCharge" in page.text


def test_saved_forecast_round_trips_on_a_daylight_saving_day(tmp_path):
    prices = synthetic_prices(days=120, start="2026-01-01")
    prices = prices[prices.index < pd.Timestamp("2026-03-28 23:00", tz="UTC")]  # next day is 29 March, 23 hours
    forecast = forecast_next_day(prices)
    assert len(forecast) == 23
    path = tmp_path / "forecast.csv"
    save_forecast(forecast, path)
    loaded = load_forecast(path)
    pd.testing.assert_frame_equal(loaded, forecast, check_freq=False)
    assert loaded.attrs["model"] == forecast.attrs["model"]
    assert load_forecast(tmp_path / "missing.csv") is None


def test_app_serves_the_saved_forecast_only_when_it_is_current(client, monkeypatch):
    api, prices = client
    real = forecast_next_day(prices)
    saved = real.assign(forecast=1.0, low=0.5, high=1.5)  # recognisable values
    saved.attrs["model"] = "ridge"
    target = daily_grid(prices).dropna().index.max() + pd.Timedelta(days=1)

    monkeypatch.setattr(web, "load_forecast", lambda: saved)
    assert web._saved_forecast(target, "ridge") is saved
    assert web._saved_forecast(target, "lightgbm") is None  # made by another model
    assert web._saved_forecast(target - pd.Timedelta(days=1), "ridge") is None  # stale

    monkeypatch.setattr(web, "store", web.Store())
    monkeypatch.setattr(web, "_cached_backtest", lambda last: web.Backtest(predictions=_tiny_backtest(prices), model_key="ridge"))
    body = api.get("/api/forecast").json()
    assert {h["forecast"] for h in body["hours"]} == {1.0}


def _tiny_backtest(prices):
    from src.day_ahead import walk_forward

    grid = daily_grid(prices)
    return walk_forward(grid, grid.dropna().index.max() - pd.Timedelta(days=20)).predictions
