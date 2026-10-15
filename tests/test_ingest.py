import pandas as pd
import pytest

from src.ingest import assemble_table, validate_hourly_table


def _sources(index: pd.DatetimeIndex):
    prices = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
    temperature = pd.DataFrame({"temperature_c": 8.0}, index=index)
    wind = pd.DataFrame({"wind_ms": 4.0}, index=index)
    return prices, temperature, wind


def test_validates_contiguous_utc_hourly_table():
    index = pd.date_range("2026-03-28T00:00Z", periods=48, freq="h", name="timestamp")
    table = assemble_table(*_sources(index))
    assert len(table) == 48
    assert table.index.tz is not None
    assert table.index.tz.zone == "UTC" if hasattr(table.index.tz, "zone") else str(table.index.tz) == "UTC"
    assert table["is_holiday"].dtype == bool


def test_rejects_missing_hour():
    index = pd.date_range("2026-01-01T00:00Z", periods=4, freq="h", name="timestamp")
    prices, temperature, wind = _sources(index)
    prices = prices.drop(index[2])
    with pytest.raises(ValueError, match="Missing required values"):
        assemble_table(prices, temperature, wind)


def test_rejects_duplicate_timestamps():
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00Z", "2026-01-01T00:00Z"], name="timestamp"
    )
    frame = pd.DataFrame(
        {"price_eur_mwh": [20.0, 21.0], "temperature_c": [8.0, 8.0], "wind_ms": [3.0, 3.0]},
        index=index,
    )
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_hourly_table(frame)


def test_resampling_fifteen_minute_prices_averages_to_hour(monkeypatch):
    from src import ingest as module

    records = [
        {"HourUTC": f"2026-01-01T00:{minute:02d}:00", "PriceArea": "SE4", "SpotPriceEUR": value}
        for minute, value in ((0, 10), (15, 20), (30, 30), (45, 40))
    ]

    def fake_download(dataset, start, end):
        return pd.DataFrame(records)

    monkeypatch.setattr(module, "_download_eds_dataset", fake_download)
    result = module.fetch_se4_prices("2026-01-01", "2026-01-02")
    assert result.iloc[0, 0] == 25
    assert len(result) == 1
