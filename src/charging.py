"""Shared EV charging logic: pick the cheapest hours inside any charging window.

Unlike the first planners, the window may cross midnight (for example plug in at
18:00 and leave at 07:00), which is how most home charging actually happens.
All functions work on an hourly price series with a timezone-aware index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

LOCAL_TIMEZONE = "Europe/Stockholm"


@dataclass(frozen=True)
class ChargeRequest:
    energy_kwh: float  # energy to add to the battery
    charger_kw: float = 7.2
    efficiency: float = 0.90

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_kwh) or self.energy_kwh <= 0:
            raise ValueError("Energy to add must be greater than zero.")
        if not math.isfinite(self.charger_kw) or self.charger_kw <= 0:
            raise ValueError("Charger power must be greater than zero.")
        if not 0 < self.efficiency <= 1:
            raise ValueError("Efficiency must be greater than 0 and at most 1.")

    @property
    def battery_kwh_per_hour(self) -> float:
        return self.charger_kw * self.efficiency

    @property
    def hours_needed(self) -> int:
        return math.ceil(self.energy_kwh / self.battery_kwh_per_hour - 1e-12)


def energy_from_soc(capacity_kwh: float, current_soc: float, target_soc: float) -> float:
    if capacity_kwh <= 0:
        raise ValueError("Battery capacity must be greater than zero.")
    if not 0 <= current_soc < target_soc <= 100:
        raise ValueError("State of charge must satisfy 0 <= current < target <= 100.")
    return capacity_kwh * (target_soc - current_soc) / 100


def charging_window(
    plug_in: pd.Timestamp, departure: pd.Timestamp, prices: pd.Series
) -> pd.Series:
    """Return the hourly prices that start inside [plug_in, departure)."""
    if departure <= plug_in:
        raise ValueError("Departure must be later than plug-in time.")
    window = prices.loc[(prices.index >= plug_in) & (prices.index < departure)]
    return window.dropna()


def _allocate(selected_index: pd.Index, request: ChargeRequest) -> pd.Series:
    """Fill the chosen hours in time order; the last hour may be partial."""
    battery_kwh = pd.Series(0.0, index=selected_index.sort_values())
    remaining = request.energy_kwh
    for stamp in battery_kwh.index:
        amount = min(request.battery_kwh_per_hour, remaining)
        battery_kwh[stamp] = amount
        remaining -= amount
    return battery_kwh


def plan(
    window_prices: pd.Series,
    request: ChargeRequest,
    *,
    strategy: str = "cheapest",
    ranking_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a charging schedule for one window.

    strategy="cheapest": charge in the hours with the lowest ranking price.
    strategy="immediate": charge as soon as the car is plugged in (the usual default).
    ranking_prices lets a forecast choose the hours while costs use window_prices.
    """
    if len(window_prices) < request.hours_needed:
        max_kwh = len(window_prices) * request.battery_kwh_per_hour
        raise ValueError(
            f"The window has {len(window_prices)} hourly slots and can add at most "
            f"{max_kwh:.1f} kWh with these charger settings."
        )
    if strategy == "immediate":
        chosen = window_prices.sort_index().index[: request.hours_needed]
    elif strategy == "cheapest":
        ranking = window_prices if ranking_prices is None else ranking_prices.reindex(window_prices.index)
        if ranking.isna().any():
            raise ValueError("Ranking prices are missing for part of the window.")
        # Stable tie-break on time so equal prices give a reproducible plan.
        ordered = ranking.to_frame("p").assign(t=range(len(ranking))).sort_values(["p", "t"])
        chosen = ordered.index[: request.hours_needed]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    battery = _allocate(chosen, request)
    schedule = pd.DataFrame({"battery_kwh": battery})
    schedule["grid_kwh"] = schedule["battery_kwh"] / request.efficiency
    schedule["price_eur_mwh"] = window_prices.reindex(schedule.index).astype(float)
    schedule["cost_eur"] = schedule["price_eur_mwh"] * schedule["grid_kwh"] / 1000
    return schedule


def add_household_cost(
    schedule: pd.DataFrame,
    *,
    sek_per_eur: float,
    retail_adder_sek_kwh: float = 0.0,
    grid_fee_sek_kwh: float = 0.0,
    energy_tax_sek_kwh: float = 0.0,
    vat_percent: float = 0.0,
) -> pd.DataFrame:
    """Add a variable household-cost column in SEK (fixed monthly fees excluded)."""
    if sek_per_eur <= 0:
        raise ValueError("SEK per EUR must be greater than zero.")
    if min(retail_adder_sek_kwh, grid_fee_sek_kwh, energy_tax_sek_kwh) < 0:
        raise ValueError("Tariff add-ons cannot be negative.")
    if not 0 <= vat_percent <= 100:
        raise ValueError("VAT must be between 0 and 100%.")
    result = schedule.copy()
    spot_sek_kwh = result["price_eur_mwh"] * sek_per_eur / 1000
    per_kwh = spot_sek_kwh + retail_adder_sek_kwh + grid_fee_sek_kwh + energy_tax_sek_kwh
    result["total_sek"] = result["grid_kwh"] * per_kwh * (1 + vat_percent / 100)
    return result
