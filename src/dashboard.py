"""Serve a lightweight local web dashboard for CalmCharge's live EV planner.

Start with: python -m src.dashboard
Then open http://127.0.0.1:8765 in a browser.
"""

from datetime import date, datetime, timedelta
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import math
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingest import fetch_se4_prices


HOST = "127.0.0.1"
PORT = 8765
LOCAL_TIMEZONE = "Europe/Stockholm"


def tomorrow_local() -> str:
    return (
        datetime.now(ZoneInfo(LOCAL_TIMEZONE)).date() + timedelta(days=1)
    ).isoformat()


def defaults() -> dict[str, str]:
    return {
        "date": tomorrow_local(),
        "start_hour": "18",
        "end_hour": "23",
        "battery_capacity_kwh": "60",
        "current_soc": "30",
        "target_soc": "50",
        "charger_kw": "7.2",
        "efficiency": "0.90",
        "sek_per_eur": "",
        "retail_adder_sek_kwh": "0",
        "grid_fee_sek_kwh": "0",
        "energy_tax_sek_kwh": "0",
        "vat_percent": "0",
    }


def make_plan(values: dict[str, str]) -> dict:
    target_date = date.fromisoformat(values["date"])
    start_hour = int(values["start_hour"])
    end_hour = int(values["end_hour"])
    capacity = float(values["battery_capacity_kwh"])
    current_soc = float(values["current_soc"])
    target_soc = float(values["target_soc"])
    charger_kw = float(values["charger_kw"])
    efficiency = float(values["efficiency"])
    retail = float(values["retail_adder_sek_kwh"])
    grid_fee = float(values["grid_fee_sek_kwh"])
    energy_tax = float(values["energy_tax_sek_kwh"])
    vat_percent = float(values["vat_percent"])
    fx_text = values["sek_per_eur"].strip()
    sek_per_eur = float(fx_text) if fx_text else None

    numeric_values = [capacity, current_soc, target_soc, charger_kw, efficiency,
                      retail, grid_fee, energy_tax, vat_percent]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Enter valid finite numbers in all fields.")
    if not 0 <= start_hour < 24 or not 1 <= end_hour <= 24 or end_hour <= start_hour:
        raise ValueError("Choose a valid same-day charging window.")
    if capacity <= 0 or not 0 <= current_soc < target_soc <= 100:
        raise ValueError("Check the battery capacity and SoC values; target must exceed current SoC.")
    if charger_kw <= 0 or not 0 < efficiency <= 1:
        raise ValueError("Charger power must be positive and efficiency must be from 0 to 1.")
    if min(retail, grid_fee, energy_tax) < 0 or not 0 <= vat_percent <= 100:
        raise ValueError("Tariff add-ons cannot be negative and VAT must be between 0 and 100%.")
    if sek_per_eur is not None and (not math.isfinite(sek_per_eur) or sek_per_eur <= 0):
        raise ValueError("SEK per EUR must be a positive number.")
    has_tariff_values = any([retail, grid_fee, energy_tax, vat_percent])
    if has_tariff_values and sek_per_eur is None:
        raise ValueError("Enter SEK per EUR to include SEK tariff values or VAT.")

    needed_kwh = capacity * (target_soc - current_soc) / 100
    battery_per_hour = charger_kw * efficiency
    needed_hours = math.ceil(needed_kwh / battery_per_hour - 1e-12)

    local_start = pd.Timestamp(target_date, tz=LOCAL_TIMEZONE)
    local_end = local_start + pd.DateOffset(days=1)
    prices = fetch_se4_prices(local_start.tz_convert("UTC"), local_end.tz_convert("UTC"))
    if prices.empty:
        raise ValueError("No day-ahead prices are published for this date yet. Try again later.")
    prices.index = prices.index.tz_convert(LOCAL_TIMEZONE)
    window = prices.loc[
        (prices.index.date == target_date)
        & (prices.index.hour >= start_hour)
        & (prices.index.hour < end_hour)
    ].dropna(subset=["price_eur_mwh"])
    expected_slots = end_hour - start_hour
    if len(window) < expected_slots:
        raise ValueError(
            f"Only {len(window)} of {expected_slots} hourly prices are available for that window."
        )
    if needed_hours > len(window):
        max_kwh = len(window) * battery_per_hour
        raise ValueError(f"The chosen window can add at most {max_kwh:.2f} kWh to the battery.")

    selected = window.nsmallest(needed_hours, "price_eur_mwh").copy()
    remaining = needed_kwh
    rows = []
    for timestamp, row in selected.iterrows():
        battery_kwh = min(battery_per_hour, remaining)
        grid_kwh = battery_kwh / efficiency
        price = float(row["price_eur_mwh"])
        market_cost_eur = price * grid_kwh / 1000
        total_sek = None
        if sek_per_eur is not None:
            spot_sek_kwh = price * sek_per_eur / 1000
            pre_vat = grid_kwh * (spot_sek_kwh + retail + grid_fee + energy_tax)
            total_sek = pre_vat * (1 + vat_percent / 100)
        rows.append({
            "time": timestamp.strftime("%Y-%m-%d %H:%M %Z"),
            "price": price,
            "battery_kwh": battery_kwh,
            "grid_kwh": grid_kwh,
            "market_cost_eur": market_cost_eur,
            "total_sek": total_sek,
        })
        remaining -= battery_kwh

    rows.sort(key=lambda item: item["time"])
    return {
        "date": target_date.isoformat(),
        "window": f"{start_hour:02d}:00–{end_hour:02d}:00",
        "needed_kwh": needed_kwh,
        "grid_kwh": sum(row["grid_kwh"] for row in rows),
        "rows": rows,
        "sek_per_eur": sek_per_eur,
        "total_eur": sum(row["market_cost_eur"] for row in rows),
        "total_sek": (
            sum(row["total_sek"] for row in rows) if sek_per_eur is not None else None
        ),
    }


def page(values: dict[str, str], result: dict | None = None, error: str | None = None) -> str:
    safe = {key: escape(str(value), quote=True) for key, value in values.items()}
    start_options = "".join(
        f'<option value="{hour}" {"selected" if values["start_hour"] == str(hour) else ""}>{hour:02d}:00</option>'
        for hour in range(24)
    )
    end_options = "".join(
        f'<option value="{hour}" {"selected" if values["end_hour"] == str(hour) else ""}>{hour:02d}:00</option>'
        for hour in range(1, 25)
    )
    tariff_mode = result is not None and result["sek_per_eur"] is not None
    result_html = ""
    if error:
        result_html = f'<section class="notice error">{escape(error)}</section>'
    elif result:
        rows = "".join(
            "<tr>"
            f"<td>{escape(row['time'])}</td>"
            f"<td>{row['price']:.2f}</td>"
            f"<td>{row['battery_kwh']:.2f}</td>"
            f"<td>{row['grid_kwh']:.2f}</td>"
            + (
                f"<td>{row['total_sek']:.2f} SEK</td>"
                if tariff_mode
                else f"<td>{row['market_cost_eur']:.3f} EUR</td>"
            )
            + "</tr>"
            for row in result["rows"]
        )
        total = (
            f"{result['total_sek']:.2f} SEK estimated variable total"
            if tariff_mode
            else f"{result['total_eur']:.3f} EUR market-energy estimate"
        )
        result_html = f"""
        <section class="result">
          <div class="result-top">
            <div><p class="eyebrow">Recommended schedule</p><h2>{escape(result['date'])} · {escape(result['window'])}</h2></div>
            <div class="total">{escape(total)}</div>
          </div>
          <p>Charge {result['needed_kwh']:.2f} kWh into the battery using {result['grid_kwh']:.2f} kWh from the grid.</p>
          <div class="table-wrap"><table>
            <thead><tr><th>Local start</th><th>Market EUR/MWh</th><th>Battery kWh</th><th>Grid kWh</th><th>{'Estimated cost'}</th></tr></thead>
            <tbody>{rows}</tbody>
          </table></div>
          <p class="fine">Variable energy estimate only. Fixed monthly charges and vehicle/charger safety controls are not included. Check the assumptions and schedule before use.</p>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CalmCharge · EV charging planner</title>
<style>
:root {{ color-scheme: light; --ink:#182525; --muted:#617170; --line:#dce5e1; --green:#136b54; --pale:#eef7f2; --card:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f4f7f5; color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1060px; margin:36px auto; padding:0 20px 48px; }}
header {{ display:flex; align-items:center; gap:14px; margin-bottom:26px; }}
.logo {{ display:grid; place-items:center; width:48px; height:48px; background:var(--green); color:#fff; border-radius:15px; font-size:25px; }}
h1,h2,p {{ margin-top:0; }}
h1 {{ font-size:27px; margin-bottom:2px; }}
.sub {{ color:var(--muted); margin:0; }}
.layout {{ display:grid; grid-template-columns:minmax(300px, 0.9fr) minmax(440px, 1.25fr); gap:20px; align-items:start; }}
.card,.result {{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:24px; box-shadow:0 8px 24px #1b3b2b0a; }}
h2 {{ font-size:20px; margin-bottom:8px; }}
h3 {{ font-size:15px; margin:22px 0 10px; }}
.fields {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
label {{ display:block; font-size:13px; color:var(--muted); font-weight:600; }}
input,select {{ width:100%; height:42px; border:1px solid #cbd8d2; border-radius:9px; background:#fff; color:var(--ink); padding:8px 10px; margin-top:5px; font:inherit; }}
button {{ border:0; border-radius:10px; background:var(--green); color:#fff; font-weight:700; padding:12px 18px; margin-top:20px; cursor:pointer; font:inherit; }}
button:hover {{ background:#0c5844; }}
.hint,.fine {{ color:var(--muted); font-size:12px; }}
.hint {{ margin:14px 0 0; }}
.notice {{ padding:15px 17px; border-radius:12px; background:#fff7e5; border:1px solid #f2d595; margin-bottom:16px; }}
.error {{ background:#fff0ee; border-color:#efc2bb; }}
.result {{ min-height:180px; }}
.result-top {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.08em; color:var(--green); font-size:11px; font-weight:800; margin-bottom:4px; }}
.total {{ background:var(--pale); color:var(--green); border-radius:12px; padding:10px 13px; font-weight:750; text-align:right; }}
.table-wrap {{ overflow-x:auto; margin-top:18px; }}
table {{ border-collapse:collapse; width:100%; white-space:nowrap; font-size:13px; }}
th,td {{ text-align:right; padding:10px 8px; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-weight:650; }}
.disclaimer {{ margin-top:18px; color:var(--muted); font-size:12px; }}
@media(max-width:800px) {{ .layout {{ grid-template-columns:1fr; }} main {{ margin-top:22px; }} }}
@media(max-width:480px) {{ .fields {{ grid-template-columns:1fr; }} .result-top {{ display:block; }} .total {{ display:inline-block; text-align:left; margin-bottom:12px; }} .card,.result {{ padding:18px; }} }}
</style>
</head>
<body><main>
<header><div class="logo">⚡</div><div><h1>CalmCharge</h1><p class="sub">Find lower-price hours for a simulated EV charge · SE4, Sweden</p></div></header>
{result_html}
<div class="layout">
<section class="card">
<h2>Charging request</h2>
<p class="sub">Uses published day-ahead prices. If tomorrow's prices are not published yet, try again later.</p>
<form method="get" action="/">
<input type="hidden" name="calculate" value="1">
<div class="fields">
<label>Swedish local date<input type="date" name="date" value="{safe['date']}" required></label>
<label>Window starts<select name="start_hour">{start_options}</select></label>
<label>Window ends (exclusive)<select name="end_hour">{end_options}</select></label>
<label>Battery capacity (kWh)<input type="number" name="battery_capacity_kwh" min="1" step="0.1" value="{safe['battery_capacity_kwh']}" required></label>
<label>Current charge (%)<input type="number" name="current_soc" min="0" max="99.9" step="0.1" value="{safe['current_soc']}" required></label>
<label>Target charge (%)<input type="number" name="target_soc" min="0.1" max="100" step="0.1" value="{safe['target_soc']}" required></label>
<label>Max AC charge power (kW)<input type="number" name="charger_kw" min="0.1" step="0.1" value="{safe['charger_kw']}" required></label>
<label>Charging efficiency<input type="number" name="efficiency" min="0.01" max="1" step="0.01" value="{safe['efficiency']}" required></label>
</div>
<h3>Optional variable tariff assumptions</h3>
<div class="fields">
<label>SEK per EUR<input type="number" name="sek_per_eur" min="0.01" step="0.01" value="{safe['sek_per_eur']}" placeholder="Leave blank for market-only"></label>
<label>Retail adder (SEK/kWh)<input type="number" name="retail_adder_sek_kwh" min="0" step="0.001" value="{safe['retail_adder_sek_kwh']}"></label>
<label>Grid fee (SEK/kWh)<input type="number" name="grid_fee_sek_kwh" min="0" step="0.001" value="{safe['grid_fee_sek_kwh']}"></label>
<label>Energy tax (SEK/kWh)<input type="number" name="energy_tax_sek_kwh" min="0" step="0.001" value="{safe['energy_tax_sek_kwh']}"></label>
<label>VAT assumption (%)<input type="number" name="vat_percent" min="0" max="100" step="0.1" value="{safe['vat_percent']}"></label>
</div>
<p class="hint">This is an educational simulation. Enter example assumptions, not a real vehicle or household bill. Fixed monthly charges are excluded.</p>
<button type="submit">Find cheapest hours</button>
</form>
</section>
<section>
{('<div class="notice">Enter a charging request to see a schedule.</div>' if not result and not error else '')}
<div class="notice">Estimated costs depend on your inputs and omit fixed charges. Verify tax/VAT treatment and vehicle limits independently.</div>
<p class="disclaimer">CalmCharge is an educational project, not a charger controller. It does not connect to a vehicle or start charging.</p>
</section>
</div>
</main></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        values = defaults()
        for key in values:
            if key in query and query[key]:
                values[key] = query[key][0]

        result = None
        error = None
        if query.get("calculate", [""])[0] == "1":
            try:
                result = make_plan(values)
            except Exception as exc:
                error = str(exc)

        document = page(values, result, error).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(document)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(document)

    def log_message(self, format: str, *args) -> None:
        # Keep the local terminal quiet when a browser requests the dashboard.
        return


def main() -> None:
    server = HTTPServer((HOST, PORT), DashboardHandler)
    print(f"CalmCharge dashboard running at http://{HOST}:{PORT}")
    print("Keep this window open. Press Ctrl+C here to stop the dashboard.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
