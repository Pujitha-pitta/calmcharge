// CalmCharge frontend: reads the JSON API and draws three charts.

const css = getComputedStyle(document.documentElement);
const color = (name) => css.getPropertyValue(name).trim();
const C = {
  lamp: color("--lamp"), idle: color("--idle"), ice: color("--ice"),
  mist: color("--mist"), line: color("--line"), frost: color("--frost"),
};
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

Chart.defaults.font.family = '"Schibsted Grotesk", system-ui, sans-serif';
Chart.defaults.color = C.mist;
Chart.defaults.borderColor = C.line;

const form = document.getElementById("settings");
const $ = (id) => document.getElementById(id);
const charts = {};

// ------------------------------------------------------------ helpers

const hhmm = (iso) => iso.slice(11, 16);
const pad = (h) => String(h).padStart(2, "0") + ":00";
const money = (value, unit) =>
  unit === "EUR" ? `€${value.toFixed(2)}` : `${value.toFixed(2)} SEK`;
const dayName = (iso, opts = { weekday: "long", day: "numeric", month: "long" }) =>
  new Date(iso.slice(0, 10) + "T12:00:00").toLocaleDateString("en-GB", opts);

function hatch(base) {
  const c = document.createElement("canvas");
  c.width = c.height = 8;
  const g = c.getContext("2d");
  g.fillStyle = base + "55";
  g.fillRect(0, 0, 8, 8);
  g.strokeStyle = base;
  g.lineWidth = 2.5;
  g.beginPath();
  g.moveTo(-2, 10); g.lineTo(10, -2);
  g.moveTo(-2, 2); g.lineTo(2, -2);
  g.moveTo(6, 10); g.lineTo(10, 6);
  g.stroke();
  return g.createPattern(c, "repeat");
}

/** Group charging hours into readable ranges like "01:00–05:00". */
function ranges(hours) {
  const out = [];
  let start = null, prev = null;
  for (const h of hours) {
    if (!h.charge) { if (start) out.push([start, prev]); start = null; continue; }
    if (!start) start = h.time;
    prev = h.time;
  }
  if (start) out.push([start, prev]);
  return out.map(([a, b]) => {
    const end = new Date(new Date(b).getTime() + 3600e3);
    const endText = end.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Stockholm" });
    return `${hhmm(a)}–${endText}`;
  });
}

function listJoin(items) {
  if (items.length <= 1) return items.join("");
  return items.slice(0, -1).join(", ") + " and " + items.at(-1);
}

function params() {
  const data = new FormData(form);
  const p = new URLSearchParams();
  for (const [k, v] of data.entries()) if (v !== "") p.set(k, v);
  return p;
}

async function getJSON(url) {
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail?.[0]?.msg || body.detail || `Request failed (${res.status})`);
  return body;
}

function draw(id, config) {
  if (charts[id]) {
    charts[id].data = config.data;
    charts[id].options = config.options;
    charts[id].update(reduceMotion ? "none" : undefined);
  } else {
    charts[id] = new Chart($(id), config);
  }
}

// ------------------------------------------------------------ tonight

async function renderPlan() {
  const note = $("forecast-note");
  let plan;
  try {
    plan = await getJSON("/api/plan?" + params());
  } catch (err) {
    $("answer").textContent = "That plan doesn't work yet.";
    $("answer-detail").textContent = "";
    note.hidden = false;
    note.className = "note error";
    note.textContent = err.message;
    return null;
  }

  const plugDate = plan.plug_in.slice(0, 10);
  const today = new Date().toLocaleDateString("en-CA", { timeZone: "Europe/Stockholm" });
  const tomorrow = new Date(Date.now() + 864e5).toLocaleDateString("en-CA", { timeZone: "Europe/Stockholm" });
  const when = plugDate === today ? "tonight" : plugDate === tomorrow ? "tomorrow night" : `on ${dayName(plan.plug_in)}`;

  const spans = ranges(plan.hours);
  $("answer").innerHTML = `Charge ${listJoin(spans.map((s) => `<span class="time">${s}</span>`))} ${when}.`;

  const saving = plan.naive_cost - plan.smart_cost;
  const pct = plan.naive_cost > 0 ? Math.round((saving / plan.naive_cost) * 100) : 0;
  $("answer-detail").innerHTML =
    `That adds ${plan.energy_kwh.toFixed(1)} kWh for <strong>${money(plan.smart_cost, plan.unit)}</strong>. ` +
    (saving > 0.005
      ? `Plugging in and charging straight away from ${hhmm(plan.plug_in)} would cost ${money(plan.naive_cost, plan.unit)}, so you save <strong>${money(saving, plan.unit)} (${pct}%)</strong>.`
      : `Charging straight away costs the same here, because the cheapest hours are the first ones.`);

  if (plan.forecast_hours > 0) {
    note.hidden = false;
    note.className = "note";
    note.textContent =
      `${plan.forecast_hours} of these ${plan.hours.length} hours use the forecast, because their prices are published around 13:00 the day before. Check again after that to confirm the plan.`;
  } else {
    note.hidden = true;
  }

  const pattern = hatch(C.idle), lampPattern = hatch(C.lamp);
  draw("plan-chart", {
    type: "bar",
    data: {
      labels: plan.hours.map((h) => hhmm(h.time)),
      datasets: [
        {
          label: "Price",
          data: plan.hours.map((h) => h.price),
          backgroundColor: plan.hours.map((h) =>
            h.source === "forecast" ? (h.charge ? lampPattern : pattern) : h.charge ? C.lamp : C.idle),
          borderRadius: 5,
          borderSkipped: false,
          categoryPercentage: 0.86,
          barPercentage: 0.94,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: reduceMotion ? false : { duration: 700, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const h = plan.hours[items[0].dataIndex];
              return `${dayName(h.time, { weekday: "short" })} ${hhmm(h.time)}`;
            },
            label: (item) => {
              const h = plan.hours[item.dataIndex];
              const lines = [`${h.price.toFixed(1)} EUR/MWh${h.source === "forecast" ? " (forecast)" : ""}`];
              if (h.low != null) lines.push(`Likely ${h.low.toFixed(0)}–${h.high.toFixed(0)} EUR/MWh`);
              if (h.charge) lines.push(`Charging ${h.battery_kwh.toFixed(1)} kWh`);
              return lines;
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true } },
        y: { title: { display: true, text: "EUR/MWh" }, grid: { color: C.line }, beginAtZero: true },
      },
    },
  });
  return plan;
}

// ------------------------------------------------------------ past year

async function renderSavings(plan) {
  const p = params();
  const q = new URLSearchParams({
    plug_hour: p.get("plug_hour"), depart_hour: p.get("depart_hour"),
    energy_kwh: plan ? plan.energy_kwh : 15, charger_kw: p.get("charger_kw"), efficiency: p.get("efficiency"),
  });
  let s;
  try {
    s = await getJSON("/api/savings?" + q);
  } catch (err) {
    $("year-summary").textContent = err.message;
    return;
  }
  const t = s.totals_eur;
  const saved = t.immediate - t.smart_published;
  $("year-summary").innerHTML =
    `Over ${s.nights} nights (${dayName(s.first_night, { day: "numeric", month: "short", year: "numeric" })} to ` +
    `${dayName(s.last_night, { day: "numeric", month: "short", year: "numeric" })}), charging ${s.energy_kwh_per_night.toFixed(1)} kWh ` +
    `between ${s.window.replace("-", " and ")} would have cost <strong>€${t.smart_published.toFixed(0)}</strong> in the cheapest hours, ` +
    `against €${t.immediate.toFixed(0)} by plugging in and charging straight away. That's <strong class="lamp">€${saved.toFixed(0)} saved, ` +
    `${s.savings_pct.smart_published.toFixed(0)}% less</strong>. ` +
    `Locking the plan in at 11:00 using only the forecast still saved ${s.savings_pct.smart_forecast.toFixed(0)}%.`;
  $("year-footnote").textContent =
    `Most of the saving comes from moving charging into the night. A simple rule that reuses yesterday's cheap hours saved ` +
    `${(s.savings_pct.smart_yesterday ?? 0).toFixed(0)}%; the forecast adds a further edge over it. ` +
    `Energy-only market prices; per-kWh grid fees and taxes are the same under every strategy, so they don't change the saving.`;

  const style = {
    immediate: { color: C.mist, dash: [] },
    smart_yesterday: { color: C.mist, dash: [5, 5] },
    smart_forecast: { color: C.ice, dash: [] },
    smart_published: { color: C.lamp, dash: [] },
  };
  draw("savings-chart", {
    type: "line",
    data: {
      labels: s.series.nights,
      datasets: Object.entries(s.labels).map(([key, label]) => ({
        label,
        data: s.series[key],
        borderColor: style[key].color,
        borderDash: style[key].dash,
        borderWidth: key === "smart_published" ? 3 : 2,
        pointRadius: 0,
        tension: 0.25,
      })),
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", align: "start", labels: { boxWidth: 18, boxHeight: 2, color: C.frost } },
        tooltip: { callbacks: { label: (i) => `${i.dataset.label}: €${i.parsed.y.toFixed(0)}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 7, callback(v) { return dayName(this.getLabelForValue(v), { month: "short", year: "2-digit" }); } } },
        y: { title: { display: true, text: "Cumulative cost (EUR)" }, grid: { color: C.line } },
      },
    },
  });
}

// ------------------------------------------------------------ forecast

async function renderForecast() {
  const f = await getJSON("/api/forecast");
  const baseline = Math.min(...f.metrics.filter((m) => m.method.startsWith("Same hour")).map((m) => m.mae));
  const best = f.metrics[0];
  $("ahead-title").textContent = `Forecast for ${dayName(f.date)}`;
  $("ahead-summary").innerHTML =
    `These prices aren't published yet. The ${best.method} model predicts them from prices that are already public. ` +
    `Tested day by day over ${f.eval_days} days, it was off by <strong>${best.mae.toFixed(1)} EUR/MWh</strong> on average, ` +
    `${Math.round((1 - best.mae / baseline) * 100)}% closer than assuming yesterday's prices repeat. ` +
    (f.coverage == null ? "" : `The shaded range held the real price ${Math.round(f.coverage * 100)}% of the time.`);

  $("status").textContent = `Live SE4 prices, forecast by ${f.model}`;

  const body = document.querySelector("#metrics tbody");
  body.replaceChildren(...f.metrics.map((m) => {
    const tr = document.createElement("tr");
    if (!m.method.startsWith("Same hour")) tr.className = "model";
    tr.innerHTML = `<td></td><td>${m.mae.toFixed(1)}</td>`;
    tr.firstChild.textContent = m.method;
    return tr;
  }));

  draw("forecast-chart", {
    type: "line",
    data: {
      labels: f.hours.map((h) => hhmm(h.time)),
      datasets: [
        { label: "High", data: f.hours.map((h) => h.high), borderWidth: 0, pointRadius: 0, fill: "+1", backgroundColor: C.ice + "33" },
        { label: "Low", data: f.hours.map((h) => h.low), borderWidth: 0, pointRadius: 0, fill: false },
        { label: "Forecast", data: f.hours.map((h) => h.forecast), borderColor: C.ice, borderWidth: 2.5, pointRadius: 2.5, pointBackgroundColor: C.ice, tension: 0.3 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (i) => i.dataset.label === "Forecast",
          callbacks: {
            label: (i) => {
              const h = f.hours[i.dataIndex];
              return [`${h.forecast.toFixed(1)} EUR/MWh`, `Likely ${h.low.toFixed(0)}–${h.high.toFixed(0)}`];
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: "EUR/MWh" }, grid: { color: C.line } },
      },
    },
  });
}

// ------------------------------------------------------------ wiring

function fillHours(select, selected) {
  for (let h = 0; h < 24; h++) select.add(new Option(pad(h), h, false, h === selected));
}
fillHours(form.plug_hour, 18);
fillHours(form.depart_hour, 7);
form.plug_date.value = new Date().toLocaleDateString("en-CA", { timeZone: "Europe/Stockholm" });

let timer;
async function refresh() {
  const plan = await renderPlan();
  await renderSavings(plan);
}
form.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(refresh, 250);
});
form.addEventListener("submit", (e) => e.preventDefault());

refresh();
renderForecast().catch((err) => { $("ahead-summary").textContent = err.message; });
