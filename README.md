# CalmCharge

Forecast SE4 day-ahead electricity prices and schedule EV charging for lower-cost hours.

## Phase 0 / Phase 1 status

- [x] Public GitHub repository and project scaffold
- [x] Price ingestion from Energi Data Service (legacy `Elspotprices` plus `DayAheadPrices`)
- [x] Hourly normalization in UTC
- [x] SMHI hourly temperature and wind observation ingestion
- [x] Swedish calendar features
- [x] Data validation tests and CI
- [ ] Run locally and verify live source coverage before proceeding to baselines

## Time and data decisions

All timestamps are stored as timezone-aware UTC. The output is hourly: 15-minute market intervals are averaged within their UTC hour, while hourly legacy prices remain unchanged. This makes the early forecasting and charging scheduler easier to reason about and prevents daylight-saving transitions from creating duplicate local timestamps or missing UTC hours.

Energi Data Service serves historical hourly prices through `Elspotprices` and newer 15-minute prices through `DayAheadPrices`; the ingestion code stitches the sources at 2025-10-01 and normalizes their different field names. Price values are kept in EUR/MWh. Convert to SEK only after defining and documenting an exchange-rate source and the applicable rate timing.

Weather uses SMHI historical hourly observations from the nearest available station around Malmö. These observations are for historical analysis. **Do not use observed future weather as a day-ahead model feature:** it would leak information unavailable at forecast time. For realistic forecasting, use archived weather forecasts as issued at the forecast cutoff, or leave weather out of the first price model.

## Setup

Requires Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

## Download and build the initial table

```powershell
python -m src.ingest --start 2024-10-15 --end 2026-10-15
```

The end date is exclusive. The command writes `data/processed/se4_hourly.parquet`. The pipeline fails with a clear error if a selected station does not cover the requested range or if the assembled hourly table contains duplicates or gaps; inspect SMHI station availability and choose a shorter initial range if needed.

## Data sources

- [Energi Data Service API](https://api.energidataservice.dk/) — SE4 day-ahead prices
- [SMHI Meteorological Observations API](https://opendata.smhi.se/metobs/) — historical hourly weather observations

## Project layout

```
src/ingest.py       Download, normalize, validate, and store phase-one data
tests/              Data-cleaning and UTC-grid tests
data/raw/           Downloaded source files (gitignored)
data/processed/     Generated parquet tables (gitignored)
```

## Limitations

The first table uses observed weather, not weather forecast vintages. It is not yet suitable for evaluating weather-based day-ahead forecasts. Market prices are energy-only EUR/MWh and do not include household taxes, network charges, supplier margins, charger efficiency, or SEK conversion. These will be explicit inputs in the later cost model.
