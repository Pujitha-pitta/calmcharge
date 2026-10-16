"""Plan EV charging with published SE4 day-ahead market prices.

Vehicle charge targets and variable household tariff add-ons are configurable.
The cost estimate is informational; the user must enter values from their own
vehicle and electricity contract.
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
        description="Plan an EV charge using published SE4 day-ahead prices."
    )
    parser.add_argument(
        "--date",
        default=tomorrow.isoformat(),
        help=f"Local Swedish date, YYYY-MM-DD (default: tomorrow, {tomorrow})",
    )
    parser.add_argument("--start-hour", required=True, type=int, help="Window start, 0-23")
    parser.add_argument("--end-hour", required=True, type=int, help="Window end, 1-24 (exclusive)")
    parser.add_argument("--energy-kwh", type=float, help="Battery energy to add, if not using SoC")
    parser.add_argument("--battery-capacity-kwh", type=float, help="Usable battery capacity")
    parser.add_argument("--current-soc", type=float, help="Current state of charge, percent")
    parser.add_argument("--target-soc", type=float, help="Desired state of charge, percent")
    parser.add_argument("--charger-kw", required=True, type=float, help="Maximum AC charging power")
    parser.add_argument(
        "--efficiency", type=float, default=0.90,
        help="Charging efficiency from grid to battery (default: 0.90)",
    )
    parser.add_argument(
        "--sek-per-eur", type=float,
        help="SEK per EUR to convert the spot price; required for tariff add-ons",
    )
    parser.add_argument(
        "--retail-adder-sek-kwh", type=float, default=0.0,
        help="Variable retailer markup in SEK/kWh, from your contract",
    )
    parser.add_argument(
        "--grid-fee-sek-kwh", type=float, default=0.0,
        help="Variable grid fee in SEK/kWh, from your network tariff",
    )
    parser.add_argument(
        "--energy-tax-sek-kwh", type=float, default=0.0,
        help="Energy tax in SEK/kWh; enter the rate applicable to you",
    )
    parser.add_argument(
        "--vat-percent", type=float, default=0.0,
        help="VAT percentage applied to the variable cost subtotal (default: 0)",
    )
    return parser.parse_args()


def get_target_energy(args: argparse.Namespace) -> tuple[float, str]:
    soc_values = (args.battery_capacity_kwh, args.current_soc, args.target_soc)
    any_soc_value = any(value is not None for value in soc_values)
    all_soc_values = all(value is not None for value in soc_values)

    if args.energy_kwh is not None and any_soc_value:
        raise ValueError("Use --energy-kwh OR all three battery/SoC options, not both.")
    if args.energy_kwh is None and not all_soc_values:
        raise ValueError(
            "Provide --energy-kwh, or provide --battery-capacity-kwh, "
            "--current-soc, and --target-soc together."
        )
    if args.energy_kwh is not None:
        if args.energy_kwh <= 0:
            raise ValueError("--energy-kwh must be greater than zero.")
        return args.energy_kwh, f"{args.energy_kwh:.2f} kWh requested"

    capacity = args.battery_capacity_kwh
    current_soc = args.current_soc
    target_soc = args.target_soc
    if capacity <= 0:
        raise ValueError("--battery-capacity-kwh must be greater than zero.")
    if not 0 <= current_soc < target_soc <= 100:
        raise ValueError("State of charge must satisfy 0 <= current SoC < target SoC <= 100.")
    energy = capacity * (target_soc - current_soc) / 100
    return energy, f"{capacity:.1f} kWh battery, {current_soc:.1f}% to {target_soc:.1f}%"


def main() -> None:
    args = parse_args()
    if not 0 <= args.start_hour < 24 or not 1 <= args.end_hour <= 24:
        raise ValueError("Use --start-hour from 0 to 23 and --end-hour from 1 to 24.")
    if args.end_hour <= args.start_hour:
        raise ValueError("This first version expects an end hour later than the start hour.")
    if args.charger_kw <= 0:
        raise ValueError("--charger-kw must be greater than zero.")
    if not 0 < args.efficiency <= 1:
        raise ValueError("--efficiency must be greater than 0 and at most 1.")

    target_energy, target_description = get_target_energy(args)
    tariff_addons = (
        args.retail_adder_sek_kwh,
        args.grid_fee_sek_kwh,
        args.energy_tax_sek_kwh,
    )
    if any(value < 0 for value in tariff_addons):
        raise ValueError("Tariff add-ons cannot be negative.")
    if not 0 <= args.vat_percent <= 100:
        raise ValueError("--vat-percent must be between 0 and 100.")
    has_tariff_values = any(value != 0 for value in tariff_addons) or args.vat_percent != 0
    if has_tariff_values and args.sek_per_eur is None:
        raise ValueError(
            "Provide --sek-per-eur when adding SEK tariff values or VAT. "
            "Use the conversion rate appropriate for your estimate."
        )
    if args.sek_per_eur is not None and args.sek_per_eur <= 0:
        raise ValueError("--sek-per-eur must be greater than zero.")

    try:
        target_date = date.fromisoformat(args.date)
    except ValueError as exc:
        raise ValueError("--date must use YYYY-MM-DD format.") from exc

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
    needed_hours = math.ceil(target_energy / battery_per_hour - 1e-12)
    if needed_hours > len(window):
        max_energy = len(window) * battery_per_hour
        raise ValueError(
            f"That window has {len(window)} hourly slots and can add at most "
            f"{max_energy:.2f} kWh to the battery with these settings."
        )

    selected = window.nsmallest(needed_hours, "price_eur_mwh").copy()
    remaining = target_energy
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

    if args.sek_per_eur is not None:
        selected["spot_sek_kwh"] = (
            selected["price_eur_mwh"].astype(float) * args.sek_per_eur / 1000
        )
        selected["pre_vat_sek"] = selected["grid_kwh"] * (
            selected["spot_sek_kwh"]
            + args.retail_adder_sek_kwh
            + args.grid_fee_sek_kwh
            + args.energy_tax_sek_kwh
        )
        selected["vat_sek"] = selected["pre_vat_sek"] * args.vat_percent / 100
        selected["total_sek"] = selected["pre_vat_sek"] + selected["vat_sek"]

    selected = selected.sort_index()

    print(f"Published hourly price slots received: {len(prices)}")
    print(f"Plan for {target_date} ({LOCAL_TIMEZONE}), {args.start_hour:02d}:00–{args.end_hour:02d}:00")
    print(
        f"Target: {target_energy:.2f} kWh into battery ({target_description}) | "
        f"Charger: {args.charger_kw:.2f} kW | Efficiency: {args.efficiency:.0%}"
    )
    print("\nSelected charging hours:")
    if args.sek_per_eur is None:
        print(f"{'Start time':20} {'EUR/MWh':>10} {'Battery kWh':>13} {'Grid kWh':>10} {'Market EUR':>12}")
    else:
        print(f"{'Start time':20} {'EUR/MWh':>10} {'Battery kWh':>13} {'Grid kWh':>10} {'Total SEK':>12}")
    for timestamp, row in selected.iterrows():
        cost = row["market_cost_eur"] if args.sek_per_eur is None else row["total_sek"]
        print(
            f"{timestamp.strftime('%Y-%m-%d %H:%M %Z'):20} "
            f"{float(row['price_eur_mwh']):10.2f} "
            f"{row['battery_kwh']:13.2f} {row['grid_kwh']:10.2f} "
            f"{cost:12.3f}"
        )

    if args.sek_per_eur is None:
        print(f"\nEstimated market-energy cost: {selected['market_cost_eur'].sum():.3f} EUR")
        print("Household tariff add-ons and VAT were not included.")
    else:
        print(f"\nEstimated variable total: {selected['total_sek'].sum():.2f} SEK")
        print(f"  Spot energy: {selected['market_cost_eur'].sum() * args.sek_per_eur:.2f} SEK")
        print(f"  VAT assumption: {args.vat_percent:.2f}%")
        print("Fixed monthly charges are excluded; verify tax treatment and rates against your contract.")
    print("Check the schedule against your vehicle and charger before using it.")


if __name__ == "__main__":
    main()
