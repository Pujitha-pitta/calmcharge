"""How much would smart charging have saved? A year of simulated nightly charging.

Every evening a simulated EV plugs in and must have the requested energy by
morning. Three strategies are compared on the same real SE4 prices:

* Plug in and charge: start charging immediately (what most cars do by default).
* Smart, published prices: after tomorrow's prices are published (~13:00),
  pick the cheapest hours in the window. Realistic for an evening plug-in.
* Smart, forecast: the plan is locked in at 11:00, *before* tomorrow's prices
  are published, so the overnight hours are ranked by the day-ahead forecast.
  Costs are always charged at the real prices.

Run after `python -m src.day_ahead`:  python -m src.backtest_savings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.charging import ChargeRequest

REPORT_DIR = Path("reports")
BACKTEST_CSV = REPORT_DIR / "day_ahead_backtest.csv"

STRATEGIES = {
    "immediate": "Plug in and charge",
    "smart_yesterday": "Rule: yesterday's cheap hours",
    "smart_forecast": "Smart, forecast at 11:00",
    "smart_published": "Smart, published prices",
}
NAIVE_COLUMN = "pred_Same hour, previous day"


@dataclass(frozen=True)
class Scenario:
    plug_in_hour: int = 18
    departure_hour: int = 7  # next morning
    request: ChargeRequest = field(default_factory=lambda: ChargeRequest(energy_kwh=15.0))

    def __post_init__(self) -> None:
        if not 0 <= self.plug_in_hour <= 23 or not 0 <= self.departure_hour <= 23:
            raise ValueError("Hours must be between 0 and 23.")


def hourly_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """(date, hour) rows -> one row per local hour with actual and forecast prices."""
    columns = ["actual", "forecast"] + ([NAIVE_COLUMN] if NAIVE_COLUMN in predictions else [])
    frame = predictions[columns].rename(columns={NAIVE_COLUMN: "naive"})
    dates = pd.to_datetime(frame.index.get_level_values("date"))
    hours = frame.index.get_level_values("hour").astype(int)
    frame.index = pd.DatetimeIndex(dates + pd.to_timedelta(hours, unit="h"), name="time")
    return frame.sort_index()


def window_costs(actual: np.ndarray, ranking: np.ndarray, request: ChargeRequest) -> np.ndarray:
    """Vectorised version of charging.plan for many windows at once.

    Rows are nights, columns are window hours. The cheapest hours by ``ranking``
    are chosen (ties go to the earlier hour, as in ``plan``), filled in time
    order so only the last chosen hour can be partial, and paid at ``actual``.
    """
    hours = request.hours_needed
    if hours > actual.shape[1]:
        max_kwh = actual.shape[1] * request.battery_kwh_per_hour
        raise ValueError(
            f"The window has {actual.shape[1]} hourly slots and can add at most "
            f"{max_kwh:.1f} kWh with these charger settings."
        )
    order = np.argsort(ranking, axis=1, kind="stable")[:, :hours]
    chosen = np.zeros(actual.shape, dtype=bool)
    np.put_along_axis(chosen, order, True, axis=1)
    rate = request.battery_kwh_per_hour
    delivered_before = (np.cumsum(chosen, axis=1) - chosen) * rate
    battery = np.where(chosen, np.clip(request.energy_kwh - delivered_before, 0.0, rate), 0.0)
    return (actual * battery).sum(axis=1) / request.efficiency / 1000


def simulate(predictions: pd.DataFrame, scenario: Scenario | None = None) -> pd.DataFrame:
    """Cost of every night in the backtest under each strategy (EUR, energy only)."""
    scenario = scenario or Scenario()
    hourly = hourly_frame(predictions)
    first_day = hourly.index.min().normalize()
    full = pd.date_range(first_day, hourly.index.max().normalize() + timedelta(days=1), freq="h", inclusive="left")
    hourly = hourly.reindex(full)

    length = (scenario.departure_hour - scenario.plug_in_hour) % 24 or 24
    offsets = scenario.plug_in_hour + np.arange(length)
    starts = np.arange(0, len(full), 24)
    starts = starts[starts + scenario.plug_in_hour + length <= len(full)]
    index = starts[:, None] + offsets[None, :]
    nights = full[starts]

    actual = hourly["actual"].to_numpy()[index]
    # At 11:00 on the plug-in day, that day's prices are public; later hours are unknown.
    tomorrow = offsets >= 24
    rankings = {"smart_forecast": np.where(tomorrow, hourly["forecast"].to_numpy()[index], actual)}
    if "naive" in hourly:
        rankings["smart_yesterday"] = np.where(tomorrow, hourly["naive"].to_numpy()[index], actual)

    complete = ~np.isnan(actual).any(axis=1)
    for ranking in rankings.values():
        complete &= ~np.isnan(ranking).any(axis=1)
    if not complete.any():
        raise ValueError("No complete charging windows in the backtest period.")
    actual = actual[complete]

    request = scenario.request
    immediate_rank = np.broadcast_to(np.arange(length, dtype=float), actual.shape)
    costs = {
        "immediate": window_costs(actual, immediate_rank, request),
        "smart_published": window_costs(actual, actual, request),
    }
    for key, ranking in rankings.items():
        costs[key] = window_costs(actual, ranking[complete], request)
    return pd.DataFrame(costs, index=pd.DatetimeIndex(nights[complete], name="night"))


def summarise(nights: pd.DataFrame, scenario: Scenario) -> dict:
    totals = nights.sum()
    base = totals["immediate"]
    possible = base - totals["smart_published"]
    return {
        "nights": len(nights),
        "first_night": str(nights.index.min().date()),
        "last_night": str(nights.index.max().date()),
        "energy_kwh_per_night": scenario.request.energy_kwh,
        "charger_kw": scenario.request.charger_kw,
        "efficiency": scenario.request.efficiency,
        "window": f"{scenario.plug_in_hour:02d}:00-{scenario.departure_hour:02d}:00",
        "totals_eur": {key: round(float(totals[key]), 2) for key in STRATEGIES if key in totals},
        "savings_pct": {
            key: round(float((base - totals[key]) / base * 100), 1)
            for key in STRATEGIES
            if key != "immediate" and key in totals
        },
        "forecast_share_of_possible_savings_pct": (
            round(float((base - totals["smart_forecast"]) / possible * 100), 1) if possible > 0 else None
        ),
    }


def _save_chart(nights: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "immediate": "#9ca3af",
        "smart_yesterday": "#f59e0b",
        "smart_forecast": "#60a5fa",
        "smart_published": "#1d4ed8",
    }
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for key in [k for k in STRATEGIES if k in nights]:
        cumulative = nights[key].cumsum()
        style = "--" if key == "smart_yesterday" else "-"
        ax.plot(cumulative.index, cumulative, style, color=colors[key], lw=2, label=STRATEGIES[key])
        ax.annotate(f"€{cumulative.iloc[-1]:,.0f}", (cumulative.index[-1], cumulative.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=colors[key])
    ax.set_ylabel("Cumulative energy cost (EUR)")
    ax.set_title("A year of simulated overnight EV charging on real SE4 prices", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.06)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest smart EV charging savings on SE4 prices.")
    parser.add_argument("--energy-kwh", type=float, default=15.0, help="Battery energy per night (default 15)")
    parser.add_argument("--charger-kw", type=float, default=7.2)
    parser.add_argument("--efficiency", type=float, default=0.90)
    parser.add_argument("--plug-in-hour", type=int, default=18)
    parser.add_argument("--departure-hour", type=int, default=7)
    parser.add_argument("--sek-per-eur", type=float, help="Optional: also report the saving in SEK")
    args = parser.parse_args()

    if not BACKTEST_CSV.exists():
        raise FileNotFoundError(f"Run `python -m src.day_ahead` first to create {BACKTEST_CSV}.")
    predictions = pd.read_csv(BACKTEST_CSV, index_col=["date", "hour"], parse_dates=["date"])
    scenario = Scenario(
        plug_in_hour=args.plug_in_hour,
        departure_hour=args.departure_hour,
        request=ChargeRequest(args.energy_kwh, args.charger_kw, args.efficiency),
    )
    nights = simulate(predictions, scenario)
    summary = summarise(nights, scenario)

    print(f"Simulated {summary['nights']} nights, {summary['first_night']} to {summary['last_night']}")
    print(f"{summary['energy_kwh_per_night']:.1f} kWh per night, plugged in {summary['window']}, "
          f"{summary['charger_kw']} kW charger, {summary['efficiency']:.0%} efficiency\n")
    print(f"{'Strategy':30} {'Energy cost (EUR)':>18} {'Saving':>8}")
    for key, label in STRATEGIES.items():
        if key not in summary["totals_eur"]:
            continue
        saving = summary["savings_pct"].get(key)
        saving_text = "" if saving is None else f"{saving:.1f}%"
        print(f"{label:30} {summary['totals_eur'][key]:18,.2f} {saving_text:>8}")
    if args.sek_per_eur:
        saved = summary["totals_eur"]["immediate"] - summary["totals_eur"]["smart_published"]
        print(f"\nSpot-energy saving at {args.sek_per_eur} SEK/EUR: {saved * args.sek_per_eur:,.0f} SEK")
    share = summary["forecast_share_of_possible_savings_pct"]
    if share is not None:
        print(f"\nPlanning at 11:00 with the forecast captured {share:.0f}% of the savings "
              "available once real prices are published.")
    print("Energy-only market prices; per-kWh grid fees and taxes are the same under every strategy.")

    REPORT_DIR.mkdir(exist_ok=True)
    nights.to_csv(REPORT_DIR / "savings_by_night.csv")
    (REPORT_DIR / "savings_summary.json").write_text(json.dumps(summary, indent=2))
    _save_chart(nights, REPORT_DIR / "savings_backtest.png")
    print(f"Saved results and chart under {REPORT_DIR}/")


if __name__ == "__main__":
    main()
