"""Plan EV charging with published SE4 day-ahead market prices.

The Energi Data Service API timestamps its request range in Danish local time.
The shared ingestion function handles that conversion and averages 15-minute
prices to hourly values for this planner.
"""

import argparse
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingest import fetch_se4_prices


LOCAL_TIMEZONE = "Europe/Stockholm"


def parse_args() -> argparse.Namespace:
    tomorrow = datetime.now(ZoneInfo(LOCAL_TIMEZONE)).date() + timedelta(days=1)
    parser = argparse.ArgumentParser(
        description="Plan an EV charging session using published SE4 day-ahead prices."
    )
    parser.add_argument(
        "--date",
        default=tomorrow.isoformat(),
        help=f"Local Swedish date, YYYY-MM-DD (default: tomorrow, {tomorrow})",
    )
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

    try:
        target_date = date.fromisoformat(args.date)
    except ValueError as exc:
        raise ValueError("Date must use YYYY-MM-DD format.") from exc

    local_start = pd.Timestamp(target_date, tz=LOCAL_TIMEZONE)
    local_end = local_start + pd.DateOffset(days=1)
    start_utc = local_start.tz_convert("UTC")
    end_utc = local_end.tz_convert("UTC")

    print(f"Fetching published SE4 day-ahead prices for {target_date}...")
    try:
        prices = fetch_se4_prices(start_utc, end_utc)
    except ValueError as exc:
        if "returned no SE4 price records" in str(exc):
            raise ValueError(
                "No day-ahead prices are published for that date yet. Try again "
                "after the market publishes prices, or select another date."
            ) from exc
        raise
    if prices.empty:
        raise ValueError(
            "No day-ahead prices are published for that date yet. Try again after "
            "the market publishes prices, or select another date."
        )

    prices.index = prices.index.tz_convert(LOCAL_TIMEZONE)
    window = prices.loc[
        (prices.index.date == target_date)
        & (prices.index.hour >= args.start_hour)
        & (prices.index.hour < args.end_hour)
    ].dropna(subset=["price_eur_mwh"])

    expected_slots = args.end_hour - args.start_hour
    if len(window) < expected_slots:
        raise ValueError(
            f"The API returned {len(window)} of {expected_slots} hourly prices "
            f"for this charging window. The published data may still be incomplete."
        )

    battery_per_hour = args.charger_kw * args.efficiency
    needed_hours = math.ceil(args.energy_kwh / battery_per_hour)
    if needed_hours > len(window):
        max_energy = len(window) * battery_per_hour
        raise ValueError(
            f"That window has {len(window)} hourly slots and can add at most "
            f"{max_energy:.2f} kWh to the battery with these settings."
        )

    selected = window.nsmallest(needed_hours, "price_eur_mwh").copy()
    remaining = args.energy_kwh
    battery_energy = []
    grid_energy = []
    for _timestamp, _row in selected.iterrows():
        amount = min(battery_per_hour, remaining)
        battery_energy.append(amount)
        grid_energy.append(amount / args.efficiency)
        remaining -= amount

    selected["battery_kwh"] = battery_energy
    selected["grid_kwh"] = grid_energy
    selected["market_cost_eur"] = (
        selected["price_eur_mwh"].astype(float) * selected["grid_kwh"] / 1000
    )
    selected = selected.sort_index()

    print(f"Published hourly price slots received: {len(prices)}")
    print(f"Plan for {target_date} ({LOCAL_TIMEZONE}), {args.start_hour:02d}:00–{args.end_hour:02d}:00")
    print(
        f"Target: {args.energy_kwh:.2f} kWh into battery | Charger: {args.charger_kw:.2f} kW | "
        f"Efficiency: {args.efficiency:.0%}"
    )
    print("\nSelected charging hours:")
    print(f"{'Start time':20} {'EUR/MWh':>10} {'Battery kWh':>13} {'Grid kWh':>10} {'Market EUR':>12}")
    for timestamp, row in selected.iterrows():
        print(
            f"{timestamp.strftime('%Y-%m-%d %H:%M %Z'):20} "
            f"{float(row['price_eur_mwh']):10.2f} "
            f"{row['battery_kwh']:13.2f} {row['grid_kwh']:10.2f} "
            f"{row['market_cost_eur']:12.3f}"
        )

    total_cost = selected["market_cost_eur"].sum()
    print(f"\nEstimated market-energy cost: {total_cost:.3f} EUR")
    print("This excludes taxes, network fees, supplier charges, and charger-specific limits.")
    print("Check the schedule against your vehicle and charger before using it.")


if __name__ == "__main__":
    main()
