"""Plan a cheapest-hour EV charge using a chosen historical SE4 price window.

This is a market-price demonstration, not a household bill or a live day-ahead schedule.
"""

import argparse
import math
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/se4_hourly.parquet")
PRICE_COLUMN = "price_eur_mwh"
LOCAL_TIMEZONE = "Europe/Stockholm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose the cheapest historical hours for an EV charging session."
    )
    parser.add_argument("--date", required=True, help="Local Swedish date, YYYY-MM-DD")
    parser.add_argument("--start-hour", required=True, type=int, help="Window start, 0-23")
    parser.add_argument("--end-hour", required=True, type=int, help="Window end, 1-24 (exclusive)")
    parser.add_argument("--energy-kwh", required=True, type=float, help="Energy to add to the battery")
    parser.add_argument("--charger-kw", required=True, type=float, help="Maximum charger power in kW")
    parser.add_argument(
        "--efficiency", type=float, default=0.90,
        help="Charging efficiency from grid to battery (default: 0.90)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.start_hour < 24 or not 1 <= args.end_hour <= 24:
        raise ValueError("Use start-hour from 0 to 23 and end-hour from 1 to 24.")
    if args.end_hour <= args.start_hour:
        raise ValueError("This first version expects an end hour later than the start hour.")
    if args.energy_kwh <= 0 or args.charger_kw <= 0:
        raise ValueError("Energy and charger power must be greater than zero.")
    if not 0 < args.efficiency <= 1:
        raise ValueError("Efficiency must be greater than 0 and at most 1.")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the Phase 1 ingestion command first."
        )

    try:
        local_date = pd.Timestamp(args.date).date()
    except Exception as exc:
        raise ValueError("Date must use YYYY-MM-DD format.") from exc

    frame = pd.read_parquet(DATA_PATH)
    if PRICE_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing required column: {PRICE_COLUMN}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")

    local = frame[[PRICE_COLUMN]].copy()
    local.index = local.index.tz_convert(LOCAL_TIMEZONE)
    window = local.loc[
        (local.index.date == local_date)
        & (local.index.hour >= args.start_hour)
        & (local.index.hour < args.end_hour)
    ].dropna(subset=[PRICE_COLUMN])

    if window.empty:
        available_start = local.index.min().date()
        available_end = local.index.max().date()
        raise ValueError(
            f"No prices found for that window. Available local dates run from "
            f"{available_start} to {available_end}."
        )

    battery_per_hour = args.charger_kw * args.efficiency
    needed_hours = math.ceil(args.energy_kwh / battery_per_hour)
    if needed_hours > len(window):
        max_energy = len(window) * battery_per_hour
        raise ValueError(
            f"That window has {len(window)} hourly slots and can add at most "
            f"{max_energy:.2f} kWh to the battery with these settings."
        )

    chosen = window.nsmallest(needed_hours, PRICE_COLUMN).copy()
    # Fill the requested battery energy; the final selected slot may be partial.
    remaining = args.energy_kwh
    battery_energy = []
    grid_energy = []
    for _timestamp, row in chosen.iterrows():
        amount = min(battery_per_hour, remaining)
        battery_energy.append(amount)
        grid_energy.append(amount / args.efficiency)
        remaining -= amount
    chosen["battery_kwh"] = battery_energy
    chosen["grid_kwh"] = grid_energy
    chosen["market_cost_eur"] = (
        chosen[PRICE_COLUMN].astype(float) * chosen["grid_kwh"] / 1000
    )
    chosen = chosen.sort_index()

    total_grid_kwh = chosen["grid_kwh"].sum()
    total_cost = chosen["market_cost_eur"].sum()
    print(f"Historical charging plan for {local_date} ({LOCAL_TIMEZONE})")
    print(
        f"Window: {args.start_hour:02d}:00 to {args.end_hour:02d}:00 | "
        f"Target: {args.energy_kwh:.2f} kWh into battery"
    )
    print(
        f"Charger: {args.charger_kw:.2f} kW | Efficiency: "
        f"{args.efficiency:.0%} | Grid energy: {total_grid_kwh:.2f} kWh"
    )
    print("\nSelected charging hours (shown in local time):")
    print(f"{'Start time':20} {'EUR/MWh':>10} {'Battery kWh':>13} {'Grid kWh':>10} {'Market EUR':>12}")
    for timestamp, row in chosen.iterrows():
        print(
            f"{timestamp.strftime('%Y-%m-%d %H:%M %Z'):20} "
            f"{float(row[PRICE_COLUMN]):10.2f} "
            f"{row['battery_kwh']:13.2f} {row['grid_kwh']:10.2f} "
            f"{row['market_cost_eur']:12.3f}"
        )
    print(f"\nEstimated market-energy cost: {total_cost:.3f} EUR")
    print("This excludes taxes, network fees, supplier charges, and charger-specific limits.")


if __name__ == "__main__":
    main()
