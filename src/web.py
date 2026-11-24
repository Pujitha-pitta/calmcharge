"""CalmCharge web app: a FastAPI backend serving JSON plus a single-page frontend.

Run locally:  python -m src.web      then open http://127.0.0.1:8000
On Vercel the same `app` runs as a function (see pyproject.toml) and the
frontend in public/ is served from the CDN.

Startup data comes from the newest of: the local price cache, or the snapshot
committed in data/snapshot/. Only prices published since then are downloaded.
"""

from __future__ import annotations

import threading
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.backtest_savings import STRATEGIES, Scenario, simulate, summarise
from src.charging import LOCAL_TIMEZONE, ChargeRequest, add_household_cost, energy_from_soc, plan
from src.day_ahead import (
    PRICE_CACHE,
    PRICE_COLUMN,
    REPORT_DIR,
    SNAPSHOT_BACKTEST,
    SNAPSHOT_PRICES,
    Backtest,
    daily_grid,
    forecast_next_day,
    load_prices,
    walk_forward,
)

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "public"
BACKTEST_CACHE = REPORT_DIR / "day_ahead_backtest.csv"
PRICE_REFRESH_SECONDS = 60 * 60
BACKTEST_DAYS = 365
# The backtest is a year of history: a few days' lag doesn't change it meaningfully,
# and rerunning it on every cold start would be slow.
BACKTEST_MAX_AGE_DAYS = 21


# ------------------------------------------------------------------ state


class Store:
    """Prices, backtest and forecast, refreshed at most once an hour."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.loaded_at = 0.0
        self.prices: pd.Series | None = None
        self.backtest: Backtest | None = None
        self.forecast: pd.DataFrame | None = None

    def get(self) -> "Store":
        with self.lock:
            if self.prices is None or clock.time() - self.loaded_at > PRICE_REFRESH_SECONDS:
                self._refresh()
        return self

    def _refresh(self) -> None:
        prices = _load_and_top_up_prices()
        grid = daily_grid(prices)
        last_complete = grid.dropna().index.max()
        backtest = _cached_backtest(last_complete)
        if backtest is None:
            backtest = walk_forward(grid, last_complete - timedelta(days=BACKTEST_DAYS - 1))
            try:
                backtest.predictions.to_csv(BACKTEST_CACHE)
            except OSError:
                pass
        self.prices = prices
        self.backtest = backtest
        self.forecast = forecast_next_day(prices, backtest)
        self.loaded_at = clock.time()


def _load_and_top_up_prices() -> pd.Series:
    from src.ingest import fetch_se4_prices

    sources = []
    try:
        sources.append(load_prices())
    except FileNotFoundError:
        pass
    if SNAPSHOT_PRICES.exists():
        sources.append(load_prices(path=Path("missing.parquet"), cache=SNAPSHOT_PRICES))
    history = max(sources, key=lambda s: s.index.max()) if sources else pd.Series(dtype=float)
    now = pd.Timestamp.now(tz="UTC")
    start = history.index.max() - timedelta(days=2) if not history.empty else now - timedelta(days=BACKTEST_DAYS + 200)
    try:
        recent = fetch_se4_prices(start, now.normalize() + timedelta(days=2))[PRICE_COLUMN].astype(float)
    except Exception:
        if history.empty:
            raise
        recent = pd.Series(dtype=float)  # offline: keep using the cache
    prices = pd.concat([history, recent])
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    try:
        PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        prices.to_frame(PRICE_COLUMN).to_csv(PRICE_CACHE)
    except OSError:
        pass
    return prices


def _cached_backtest(last_complete: pd.Timestamp) -> Backtest | None:
    """Reuse the newest saved backtest (local cache or committed snapshot) if it is recent."""
    frames = [
        pd.read_csv(path, index_col=["date", "hour"], parse_dates=["date"])
        for path in (BACKTEST_CACHE, SNAPSHOT_BACKTEST)
        if path.exists()
    ]
    if not frames:
        return None
    frame = max(frames, key=lambda f: f.index.get_level_values("date").max())
    age = last_complete - frame.index.get_level_values("date").max()
    if age > timedelta(days=BACKTEST_MAX_AGE_DAYS) or age < timedelta(0):
        return None
    errors = {c: (frame[c] - frame["actual"]).abs().mean() for c in frame if c in ("pred_LightGBM", "pred_Ridge regression")}
    best = min(errors, key=errors.get) if errors else "pred_Ridge regression"
    return Backtest(predictions=frame, model_key="lightgbm" if best == "pred_LightGBM" else "ridge")


store = Store()
app = FastAPI(title="CalmCharge", docs_url="/api/docs", redoc_url=None)


def _num(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 3)


# --------------------------------------------------------------------- API


@app.get("/api/plan")
def api_plan(
    plug_date: date | None = None,
    plug_hour: int = Query(18, ge=0, le=23),
    depart_hour: int = Query(7, ge=0, le=23),
    capacity_kwh: float = Query(60, gt=0, le=250),
    soc_from: float = Query(30, ge=0, le=100),
    soc_to: float = Query(80, ge=0, le=100),
    charger_kw: float = Query(7.2, gt=0, le=50),
    efficiency: float = Query(0.9, gt=0, le=1),
    sek_per_eur: float = Query(0, ge=0),
    retail_sek_kwh: float = Query(0, ge=0),
    grid_fee_sek_kwh: float = Query(0, ge=0),
    energy_tax_sek_kwh: float = Query(0, ge=0),
    vat_percent: float = Query(0, ge=0, le=100),
):
    data = store.get()
    plug_date = plug_date or pd.Timestamp.now(tz=LOCAL_TIMEZONE).date()
    plug_in = pd.Timestamp(datetime.combine(plug_date, time(plug_hour))).tz_localize(LOCAL_TIMEZONE)
    departure = pd.Timestamp(datetime.combine(plug_date, time(depart_hour))).tz_localize(LOCAL_TIMEZONE)
    if departure <= plug_in:
        departure += pd.Timedelta(days=1)
    hours = pd.date_range(plug_in, departure, freq="h", inclusive="left")

    window = pd.DataFrame({"published": data.prices.tz_convert(LOCAL_TIMEZONE).reindex(hours)})
    window = window.join(data.forecast[["forecast", "low", "high"]])
    window["price"] = window["published"].fillna(window["forecast"])
    if window["price"].isna().any():
        raise HTTPException(
            422,
            "Prices for this window aren't published or forecastable yet. Choose a plug-in date "
            "no more than a day ahead.",
        )
    try:
        request = ChargeRequest(energy_from_soc(capacity_kwh, soc_from, soc_to), charger_kw, efficiency)
        smart = plan(window["price"], request)
        naive = plan(window["price"], request, strategy="immediate")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    unit, cost_col = "EUR", "cost_eur"
    if sek_per_eur > 0:
        tariff = dict(sek_per_eur=sek_per_eur, retail_adder_sek_kwh=retail_sek_kwh,
                      grid_fee_sek_kwh=grid_fee_sek_kwh, energy_tax_sek_kwh=energy_tax_sek_kwh,
                      vat_percent=vat_percent)
        smart, naive = add_household_cost(smart, **tariff), add_household_cost(naive, **tariff)
        unit, cost_col = "SEK", "total_sek"

    rows = []
    for stamp, row in window.iterrows():
        forecasted = pd.isna(row["published"])
        rows.append({
            "time": stamp.isoformat(),
            "price": _num(row["price"]),
            "source": "forecast" if forecasted else "published",
            "low": _num(row["low"]) if forecasted else None,
            "high": _num(row["high"]) if forecasted else None,
            "battery_kwh": _num(smart["battery_kwh"].get(stamp, 0.0)),
            "charge": stamp in smart.index,
        })
    return {
        "plug_in": plug_in.isoformat(),
        "departure": departure.isoformat(),
        "energy_kwh": round(request.energy_kwh, 2),
        "hours_needed": request.hours_needed,
        "unit": unit,
        "smart_cost": _num(smart[cost_col].sum()),
        "naive_cost": _num(naive[cost_col].sum()),
        "forecast_hours": int(window["published"].isna().sum()),
        "hours": rows,
    }


@app.get("/api/forecast")
def api_forecast():
    data = store.get()
    frame = data.backtest.predictions
    errors = frame.filter(like="pred_").sub(frame["actual"], axis=0).abs().mean()
    metrics = [{"method": k.removeprefix("pred_"), "mae": round(float(v), 2)} for k, v in errors.sort_values().items()]
    inside = frame.dropna(subset=["low", "high"])
    coverage = ((inside["actual"] >= inside["low"]) & (inside["actual"] <= inside["high"])).mean()
    days = frame.index.get_level_values("date")
    forecast = data.forecast
    return {
        "date": forecast.index[0].date().isoformat(),
        "model": "LightGBM" if forecast.attrs.get("model") == "lightgbm" else "Ridge regression",
        "hours": [
            {"time": t.isoformat(), "forecast": _num(r.forecast), "low": _num(r.low), "high": _num(r.high)}
            for t, r in forecast.iterrows()
        ],
        "metrics": metrics,
        "coverage": _num(coverage),  # None until the backtest is long enough for ranges
        "eval_days": int(days.nunique()),
        "eval_start": days.min().date().isoformat(),
        "eval_end": days.max().date().isoformat(),
    }


@app.get("/api/savings")
def api_savings(
    plug_hour: int = Query(18, ge=0, le=23),
    depart_hour: int = Query(7, ge=0, le=23),
    energy_kwh: float = Query(15, gt=0, le=250),
    charger_kw: float = Query(7.2, gt=0, le=50),
    efficiency: float = Query(0.9, gt=0, le=1),
):
    data = store.get()
    try:
        scenario = Scenario(plug_in_hour=plug_hour, departure_hour=depart_hour,
                            request=ChargeRequest(energy_kwh, charger_kw, efficiency))
        nights = simulate(data.backtest.predictions, scenario)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    summary = summarise(nights, scenario)
    cumulative = nights.cumsum()
    weekly = cumulative.iloc[::7]
    if weekly.index[-1] != cumulative.index[-1]:
        weekly = pd.concat([weekly, cumulative.iloc[[-1]]])
    return {
        **summary,
        "labels": {k: v for k, v in STRATEGIES.items() if k in nights},
        "series": {
            "nights": [d.date().isoformat() for d in weekly.index],
            **{key: [round(float(v), 2) for v in weekly[key]] for key in nights.columns},
        },
    }


# ------------------------------------------------------------------ pages

@app.get("/", include_in_schema=False)
def index():
    # On Vercel the CDN serves public/index.html before requests reach the app.
    page = PUBLIC_DIR / "index.html"
    return FileResponse(page) if page.exists() else RedirectResponse("/index.html")


# Locally, serve the frontend files too (registered last so /api routes win).
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")


def main() -> None:
    import uvicorn

    print("Preparing prices and forecast...")
    store.get()
    print("CalmCharge is running at http://127.0.0.1:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
