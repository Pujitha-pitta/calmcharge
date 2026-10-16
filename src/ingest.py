"""Download and normalize the phase-one CalmCharge dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import holidays
import pandas as pd
import requests

EDS_URL = "https://api.energidataservice.dk/dataset"
SMHI_URL = "https://opendata-download-metobs.smhi.se/api/version/latest"
PRICE_CUTOVER = pd.Timestamp("2025-10-01T00:00:00Z")
MALMO_LAT, MALMO_LON = 55.6050, 13.0038
DATA_PATH = Path("data/processed/se4_hourly.parquet")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CalmCharge/0.1 (open educational project)"})


def _get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = SESSION.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of {names}; received columns: {list(frame.columns)}")


def _download_eds_dataset(
    dataset: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = _get_json(
            f"{EDS_URL}/{dataset}",
            params={
                "start": start.strftime("%Y-%m-%dT%H:%M"),
                "end": end.strftime("%Y-%m-%dT%H:%M"),
                "filter": json.dumps({"PriceArea": ["SE4"]}),
                "limit": 10_000,
                "offset": offset,
            },
        )
        batch = payload.get("records", [])
        if not batch:
            break
        records.extend(batch)
        if len(batch) < 10_000:
            break
        offset += len(batch)
    return pd.DataFrame.from_records(records)


def fetch_se4_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch SE4 spot prices over [start, end), returning hourly UTC EUR/MWh."""
    start_utc, end_utc = _as_utc(start), _as_utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    pieces: list[pd.DataFrame] = []
    if start_utc < PRICE_CUTOVER:
        old_end = min(end_utc, PRICE_CUTOVER)
        pieces.append(_download_eds_dataset("Elspotprices", start_utc, old_end))
    if end_utc > PRICE_CUTOVER:
        new_start = max(start_utc, PRICE_CUTOVER)
        pieces.append(_download_eds_dataset("DayAheadPrices", new_start, end_utc))

    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if raw.empty:
        raise ValueError("Energi Data Service returned no SE4 price records")

    time_col = _first_present(raw, ("HourUTC", "TimeUTC", "MinutesUTC"))
    price_col = _first_present(
        raw,
        ("SpotPriceEUR", "DayAheadPriceEUR"),
    )
    if "PriceArea" in raw:
        raw = raw.loc[raw["PriceArea"].eq("SE4")].copy()
    result = raw[[time_col, price_col]].copy()
    result.columns = ["timestamp", "price"]
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result["price"] = pd.to_numeric(result["price"], errors="coerce")
    result = result.dropna(subset=["timestamp", "price"])

    # The newer market is 15-minute; averaging four intervals gives one hourly
    # energy price. Legacy hourly observations remain unchanged by this operation.
    hourly = result.set_index("timestamp")["price"].resample("h").mean()
    hourly = hourly.rename("price_eur_mwh").to_frame()
    hourly.index.name = "timestamp"
    return hourly.loc[(hourly.index >= start_utc) & (hourly.index < end_utc)]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        delta_lon / 2
    ) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def find_active_stations(parameter: int, limit: int = 8) -> list[dict[str, Any]]:
    """Return nearby active stations with archive/recent observation periods."""
    metadata = _get_json(f"{SMHI_URL}/parameter/{parameter}.json")
    candidates = [
        station
        for station in metadata.get("station", [])
        if station.get("active") is True
        and station.get("latitude") is not None
        and station.get("longitude") is not None
    ]
    candidates.sort(
        key=lambda station: _haversine_km(
            MALMO_LAT,
            MALMO_LON,
            float(station["latitude"]),
            float(station["longitude"]),
        )
    )
    usable = []
    for station in candidates:
        detail = _get_json(
            f"{SMHI_URL}/parameter/{parameter}/station/{station['key']}.json"
        )
        available = {item.get("key") for item in detail.get("period", [])}
        periods = [
            key for key in ("corrected-archive", "latest-months") if key in available
        ]
        if periods:
            usable.append({**station, "_period_keys": periods})
            if len(usable) >= limit:
                break
    if not usable:
        raise ValueError(
            f"No active SMHI station has a corrected archive or latest-months period "
            f"for parameter {parameter}"
        )
    return usable


def find_nearest_station(parameter: int) -> dict[str, Any]:
    """Return the nearest active station with an observation period."""
    return find_active_stations(parameter, limit=1)[0]

def _parse_smhi_csv(text: str, station_name: str, column: str) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if len(row) >= 3
            and row[0].strip().lstrip("\ufeff").lower() in ("datum", "date")
            and ("tid" in row[1].strip().lower() or row[1].strip().lower() == "time")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Could not find the date/time header in SMHI data for {station_name}")
    header = [item.strip() for item in rows[header_index]]
    value_index = next(
        (
            index
            for index, name in enumerate(header[2:], start=2)
            if "kvalitet" not in name.lower() and name.strip()
        ),
        None,
    )
    if value_index is None:
        raise ValueError(f"SMHI response for {station_name} has no value column")
    parsed_rows = [
        (row[0], row[1], row[value_index])
        for row in rows[header_index + 1 :]
        if len(row) > value_index and row[0].strip()
    ]
    raw = pd.DataFrame(parsed_rows, columns=["date", "time", "value"])
    timestamps = pd.to_datetime(
        raw["date"].astype(str).str.strip()
        + " "
        + raw["time"].astype(str).str.strip(),
        format="mixed",
        utc=True,
        errors="coerce",
    )
    values = pd.to_numeric(raw["value"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=timestamps, name=column).dropna()
    series.index.name = "timestamp"
    series = series.loc[~series.index.isna()]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series.resample("h").mean().to_frame()


def fetch_smhi_observations(
    parameter: int, column: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch hourly observations, filling station gaps from nearby active stations."""
    stations = find_active_stations(parameter)
    expected = pd.date_range(
        start.ceil("h"), end.ceil("h"), freq="h", inclusive="left", name="timestamp"
    )
    combined = pd.DataFrame({column: pd.Series(index=expected, dtype=float)}, index=expected)
    for station in stations:
        pieces = []
        for period in station["_period_keys"]:
            url = (
                f"{SMHI_URL}/parameter/{parameter}/station/{station['key']}"
                f"/period/{period}/data.csv"
            )
            response = SESSION.get(url, timeout=120)
            response.raise_for_status()
            pieces.append(
                _parse_smhi_csv(
                    response.text, station.get("name", str(station["key"])), column
                )
            )
        station_data = pd.concat(pieces).sort_index()
        station_data = station_data[~station_data.index.duplicated(keep="last")]
        station_data = station_data.reindex(expected)
        # Keep the closest station's values and use farther stations only where
        # that station has no observation for the same UTC hour.
        combined[column] = combined[column].combine_first(station_data[column])
        if combined[column].notna().all():
            break
    if combined[column].isna().any():
        missing = int(combined[column].isna().sum())
        raise ValueError(
            f"SMHI parameter {parameter} still has {missing} missing hourly values "
            f"between {start} and {end} after checking {len(stations)} nearby active stations"
        )
    return combined

def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = result.index.tz_convert("Europe/Stockholm")
    swedish_holidays = holidays.country_holidays(
        "SE", years=range(local.min().year, local.max().year + 1)
    )
    result["hour"] = local.hour
    result["day_of_week"] = local.dayofweek
    result["month"] = local.month
    result["is_holiday"] = [stamp.date() in swedish_holidays for stamp in local]
    return result


def assemble_table(
    prices: pd.DataFrame, temperature: pd.DataFrame, wind: pd.DataFrame
) -> pd.DataFrame:
    """Join sources on an explicit continuous UTC-hour grid and reject gaps."""
    if prices.empty:
        raise ValueError("Price data is empty")
    start = prices.index.min().floor("h")
    end = prices.index.max().floor("h")
    full_index = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp")
    table = prices.reindex(full_index)
    table = table.join(temperature.reindex(full_index), how="left")
    table = table.join(wind.reindex(full_index), how="left")
    table = add_calendar_features(table)
    return validate_hourly_table(table)


def validate_hourly_table(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("The table must have a timezone-aware DatetimeIndex")
    if str(frame.index.tz) != "UTC":
        raise ValueError("The table index must be UTC")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps found")
    expected = pd.date_range(
        frame.index.min(), frame.index.max(), freq="h", tz="UTC", name=frame.index.name
    )
    if not frame.index.equals(expected):
        raise ValueError("Missing or irregular UTC hours found")
    required = ["price_eur_mwh", "temperature_c", "wind_ms"]
    missing_columns = [name for name in required if name not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    missing = frame[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing required values: {missing[missing > 0].to_dict()}")
    return frame.sort_index()


def ingest(start: str, end: str, output: Path = DATA_PATH) -> Path:
    start_utc, end_utc = _as_utc(start), _as_utc(end)
    prices = fetch_se4_prices(start, end)
    temperature = fetch_smhi_observations(1, "temperature_c", start_utc, end_utc)
    wind = fetch_smhi_observations(4, "wind_ms", start_utc, end_utc)
    table = assemble_table(prices, temperature, wind)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=(date.today() - timedelta(days=730)).isoformat())
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    print(f"Wrote {ingest(args.start, args.end, args.output)}")


if __name__ == "__main__":
    main()
