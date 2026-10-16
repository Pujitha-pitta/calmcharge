# CalmCharge

An electricity-price analytics and EV charging project for Sweden's SE4 region. The project ingests public price and weather data, explores price patterns, and compares forecasting methods before building toward a cost-aware charging scheduler.

## Progress

- [x] Repository scaffold and CI
- [x] Hourly SE4 electricity-price ingestion and normalization
- [x] Historical weather and Swedish calendar features
- [x] Data quality checks: hourly timestamps, duplicates, and missing values
- [x] Exploratory charts by hour, weekday, and month
- [x] Baseline comparisons and first ridge-regression model
- [x] Expanding-window time-series validation
- [ ] Build a practical charging recommendation and cost model

## Current experiment

The local dataset used for these results contains 17,520 hourly observations from 2024-10-15 through 2026-10-14. Each prediction is one hour ahead and uses earlier observed prices. This is not yet a day-ahead forecast.

A chronological holdout using the earliest 70% as training history and latest 30% for evaluation produced:

| Method | MAE (EUR/MWh) | RMSE (EUR/MWh) |
| --- | ---: | ---: |
| Ridge regression model | 14.41 | 22.45 |
| Yesterday, same UTC hour | 39.99 | 57.73 |
| Previous 7 days, same UTC hour average | 41.03 | 53.30 |
| Last week, same UTC hour | 48.05 | 65.79 |

The expanding-window validation then trained on progressively longer histories and evaluated three later periods:

| Validation fold | Evaluation period (UTC) | Ridge model MAE / RMSE |
| --- | --- | ---: |
| 1 | 2026-03-12 02:00 to 2026-05-23 08:00 | 12.29 / 17.19 |
| 2 | 2026-05-23 09:00 to 2026-08-03 15:00 | 13.23 / 20.57 |
| 3 | 2026-08-03 16:00 to 2026-10-14 23:00 | 17.72 / 28.29 |

Combined across the 5,206 validation hours, the ridge model scored **MAE 14.41** and **RMSE 22.50 EUR/MWh**. The baselines scored MAE 40.10 (yesterday), 41.06 (previous seven-day same-hour average), and 48.14 (last week). The ridge model led on both combined metrics in all three folds. Scores worsen in the latest fold, so this result is evidence for this dataset and period, not a guarantee of future performance.

## Explore the data and run the models

Requires Python 3.11 or newer. In PowerShell, from the repository folder:

```powershell
.\.venv\Scripts\python.exe -m src.explore
.\.venv\Scripts\python.exe -m src.evaluate_baselines
.\.venv\Scripts\python.exe -m src.train_model
.\.venv\Scripts\python.exe -m src.validate_model
```

The exploration script prints summary statistics and saves three local charts under `reports/phase2/`:

- `average_by_hour.png`
- `average_by_weekday.png`
- `average_by_month.png`

The validation script uses expanding training windows and later, unseen timestamps for evaluation. The charts and parquet data are generated locally and are not committed to the repository.

## Build the initial table

Requires Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
python -m src.ingest --start 2024-10-15 --end 2026-10-15
```

The end date is exclusive. The ingestion command writes `data/processed/se4_hourly.parquet`. Price and time data are normalized to an hourly UTC index. The source API switches to Denmark's winter time at local midnight on 2025-10-01; the ingestion code handles that source boundary when stitching its price datasets.

## Time, features, and data limitations

All stored timestamps are timezone-aware UTC. The charts group calendar patterns using Sweden's local time. The first model uses calendar fields and earlier observed prices. It does not use observed future weather, which would leak information unavailable at forecast time.

SMHI weather observations are historical analysis inputs. A realistic day-ahead weather model needs archived weather forecasts as issued at the forecast cutoff. Prices are energy-only EUR/MWh and do not include household taxes, network charges, supplier margins, charger efficiency, or SEK conversion. These need to be explicit inputs in a later EV charging cost model.

## Data sources

- [Energi Data Service API](https://api.energidataservice.dk/) — SE4 day-ahead prices
- [SMHI Meteorological Observations API](https://opendata.smhi.se/metobs/) — historical hourly weather observations

## Project layout

```
src/ingest.py                 Download, normalize, validate, and store phase-one data
src/explore.py                Summary statistics and exploratory charts
src/evaluate_baselines.py     Chronological baseline metrics
src/train_model.py            One-hour-ahead ridge-regression model
src/validate_model.py         Expanding-window time-series evaluation
tests/                        Data cleaning and UTC-grid tests
data/raw/                     Downloaded source files (gitignored)
data/processed/               Generated parquet tables (gitignored)
reports/phase2/               Generated charts (local output)
```
