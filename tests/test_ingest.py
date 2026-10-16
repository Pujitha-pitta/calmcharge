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
    assert str(table.index.tz) == "UTC"
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


def test_station_lookup_uses_station_detail_to_find_archive(monkeypatch):
    from src import ingest as module

    responses = {
        f"{module.SMHI_URL}/parameter/1.json": {
            "station": [
                {"key": "near", "name": "Malmö 2", "latitude": 55.60, "longitude": 13.00, "active": False},
                {"key": "archive", "name": "Lund", "latitude": 55.70, "longitude": 13.20, "active": True},
            ]
        },
        f"{module.SMHI_URL}/parameter/1/station/near.json": {
            "period": [{"key": "latest-hour"}]
        },
        f"{module.SMHI_URL}/parameter/1/station/archive.json": {
            "period": [{"key": "corrected-archive"}, {"key": "latest-months"}]
        },
    }
    monkeypatch.setattr(module, "_get_json", lambda url, params=None: responses[url])

    station = module.find_nearest_station(1)

    assert station["key"] == "archive"
    assert station["_period_keys"] == ["corrected-archive", "latest-months"]


def test_smhi_csv_parser_tolerates_extra_semicolon_field(monkeypatch):
    from src import ingest as module

    class FakeResponse:
        text = (
            "# metadata line\n"
            "Datum;Tid (UTC);Lufttemperatur;Kvalitet\n"
            "2026-01-01;00:00:00;8.5;G;extra\n"
        )

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        module,
        "find_active_stations",
        lambda parameter: [{"key": "test", "name": "test", "_period_keys": ["corrected-archive"]}],
    )
    monkeypatch.setattr(module.SESSION, "get", lambda *args, **kwargs: FakeResponse())

    result = module.fetch_smhi_observations(
        1,
        "temperature_c",
        pd.Timestamp("2026-01-01T00:00Z"),
        pd.Timestamp("2026-01-01T01:00Z"),
    )

    assert result.iloc[0, 0] == pytest.approx(8.5)



def test_eds_query_bounds_are_converted_from_utc_to_danish_time(monkeypatch):
    from src import ingest as module

    captured = {}

    def fake_get_json(url, params=None):
        captured["params"] = params
        return {"records": []}

    monkeypatch.setattr(module, "_get_json", fake_get_json)
    module._download_eds_dataset(
        "DayAheadPrices",
        pd.Timestamp("2025-10-01T00:00Z"),
        pd.Timestamp("2025-10-02T00:00Z"),
    )

    assert captured["params"]["start"] == "2025-10-01T02:00"
    assert captured["params"]["end"] == "2025-10-02T02:00"



def test_get_json_waits_and_retries_after_rate_limit(monkeypatch):
    from src import ingest as module

    class FakeResponse:
        def __init__(self, status_code, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.text = ""

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("HTTP error")

        def json(self):
            return self._payload

    responses = [
        FakeResponse(429, {"message": "rate limited"}, {"Retry-After": "3"}),
        FakeResponse(200, {"records": [1]}),
    ]
    waits = []
    monkeypatch.setattr(module.SESSION, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr(module.time, "sleep", waits.append)

    assert module._get_json("https://example.test") == {"records": [1]}
    assert waits == [3.0]



def test_combines_legacy_and_new_price_timestamp_columns(monkeypatch):
    from src import ingest as module

    old = pd.DataFrame(
        {
            "HourUTC": ["2025-09-30T23:00:00"],
            "PriceArea": ["SE4"],
            "SpotPriceEUR": [40.0],
        }
    )
    new = pd.DataFrame(
        {
            "TimeUTC": ["2025-10-01T00:00:00"],
            "PriceArea": ["SE4"],
            "DayAheadPriceEUR": [60.0],
        }
    )

    def fake_download(dataset, start, end):
        return old if dataset == "Elspotprices" else new

    monkeypatch.setattr(module, "_download_eds_dataset", fake_download)
    result = module.fetch_se4_prices(
        "2025-09-30T23:00Z", "2025-10-01T01:00Z"
    )

    assert list(result["price_eur_mwh"]) == [40.0, 60.0]
    assert result.index[0] == pd.Timestamp("2025-09-30T23:00Z")
    assert result.index[-1] == pd.Timestamp("2025-10-01T00:00Z")
