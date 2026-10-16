# CalmCharge

An electricity-price analytics and EV charging project for Sweden's SE4 region. The project ingests public price and weather data, explores price patterns, evaluates forecasting methods, and plans EV charging around public day-ahead prices.

## Progress

- [x] Repository scaffold and CI
- [x] Hourly SE4 electricity-price ingestion and normalization
- [x] Historical weather and Swedish calendar features
- [x] Data quality checks: hourly timestamps, duplicates, and missing values
- [x] Exploratory charts by hour, weekday, and month
- [x] Baseline comparisons and first ridge-regression model
- [x] Expanding-window time-series validation
- [x] Cheapest-hour charging planner using historical prices
- [x] Fetch published SE4 day-ahead prices for an upcoming schedule
- [x] Account for configurable battery state and variable tariff inputs
- [ ] Validate the cost estimate against an actual electricity bill

## Forecasting experiment

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

## Historical charging-plan demo

Choose a date and an available charging window from your downloaded dataset. Example: request 10 kWh into the battery between 18:00 and 23:00 on October 14, 2026, with a 7.2 kW charger and 90% efficiency:

```powershell
.\.venv\Scripts\python.exe -m src.plan_charge --date 2026-10-14 --start-hour 18 --end-hour 23 --energy-kwh 10 --charger-kw 7.2 --efficiency 0.90
```

This historical demo selects the cheapest hours in the window and estimates market-energy cost. It does not make a live recommendation.

## Upcoming day-ahead charging plan

The live planner requests published SE4 day-ahead prices from Energi Data Service. By default, it tries tomorrow's date in Sweden time. If the market has not published prices yet, wait and run the command again later.

No real car is needed for the project: use an explicitly labeled simulated EV profile. For example, the following assumes a 60 kWh usable battery, charging from 30% to 50%, a 7.2 kW AC limit, and 90% efficiency. These are example settings, not specifications for a particular vehicle:

```powershell
.\.venv\Scripts\python.exe -m src.plan_live_charge --start-hour 18 --end-hour 23 --battery-capacity-kwh 60 --current-soc 30 --target-soc 50 --charger-kw 7.2 --efficiency 0.90
```

The planner derives the requested energy from the battery capacity and state-of-charge change, then selects the cheapest hours. It reports a market-energy estimate in EUR. Optional variable tariff inputs are `--retail-adder-sek-kwh`, `--grid-fee-sek-kwh`, and `--energy-tax-sek-kwh`. To include them, provide the corresponding SEK/kWh values from a sample contract, plus `--sek-per-eur` and `--vat-percent`. Clearly label assumed values in project reports. Fixed monthly charges are excluded. Verify taxes and fees against the applicable contract before treating the estimate as a household bill.

The API request timestamps use Danish local time; the project converts them to UTC and displays the schedule in Swedish local time.

## Local web dashboard

Start a small dashboard on your computer using the same live planner and simulated EV defaults:

```powershell
.\.venv\Scripts\python.exe -m src.dashboard
```

Leave that PowerShell window open, then visit [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser. Choose a date, charging window, and simulated battery settings, then select **Find cheapest hours**. The dashboard requests published SE4 day-ahead prices, so a date may be unavailable until prices are published.

Stop the dashboard with **Ctrl+C** in its PowerShell window. It runs only on your computer and does not connect to or control a vehicle or charger. Cost results are estimates based on the entered assumptions; fixed charges are excluded.

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

The end date is exclusive. The ingestion command writes `data/processed/se4_hourly.parquet`. Price and time data are normalized to an hourly UTC index. The source API switches to newer market intervals at Denmark's local midnight on 2025-10-01; the ingestion code handles that source boundary when stitching its price datasets.

## Time, features, and data limitations

All stored timestamps are timezone-aware UTC. Charts and charging plans display Sweden's local time. The first model uses calendar fields and earlier observed prices; it does not use observed future weather, which would leak information unavailable at forecast time.

SMHI weather observations are historical analysis inputs. A realistic weather-based day-ahead model needs archived weather forecasts as issued at the forecast cutoff. Market prices are EUR/MWh and do not include household fees or taxes unless you provide variable tariff inputs; verify those inputs against your own electricity contract.

## Data sources

- [Energi Data Service API](https://api.energidataservice.dk/) — SE4 day-ahead prices
- [Energi Data Service API guide](https://energidataservice.dk/guides/api-guides) — API query rules and Danish-local timestamps
- [SMHI Meteorological Observations API](https://opendata.smhi.se/metobs/) — historical hourly weather observations

## Project layout

```
src/ingest.py                 Download, normalize, validate, and store phase-one data
src/explore.py                Summary statistics and exploratory charts
src/evaluate_baselines.py     Chronological baseline metrics
src/train_model.py            One-hour-ahead ridge-regression model
src/validate_model.py         Expanding-window time-series evaluation
src/plan_charge.py            Cheapest-hour historical charging demo
src/plan_live_charge.py       Live plan with simulated EV and tariff inputs
src/dashboard.py              Local web dashboard for the simulated live planner
tests/                        Data cleaning and UTC-grid tests
data/raw/                     Downloaded source files (gitignored)
data/processed/               Generated parquet tables (gitignored)
reports/phase2/               Generated charts (local output)
```
