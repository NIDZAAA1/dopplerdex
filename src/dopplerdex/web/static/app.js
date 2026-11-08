"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { uploadId: null, region: null, analysis: null, placeBox: null, whereMode: "continent" };

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${b.dataset.tab}`; });
  if (b.dataset.tab === "locate") setTimeout(() => map.invalidateSize(), 50);
  if (b.dataset.tab === "data") loadIngested();
}));

// ------------------------------------------------------------------ map
const map = L.map("map", { worldCopyJump: true }).setView([30, 0], 2);
const base = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri, HERE, Garmin, © OpenStreetMap", maxZoom: 16 }).addTo(map);
const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { attribution: "Esri", maxZoom: 19 });
L.control.layers({ "Dark map": base, "Satellite": sat }).addTo(map);
const layers = { region: L.layerGroup().addTo(map), results: L.layerGroup().addTo(map) };

function drawRegion(info) {
  layers.region.clearLayers();
  if (!info) return;
  const [s, n, w, e] = info.bbox;
  L.rectangle([[s, w], [n, e]], { color: "#8a98a8", weight: 1.5, dashArray: "6 6", fill: false }).bindTooltip(`search area: ${info.label}`).addTo(layers.region);
  map.fitBounds([[s, w], [n, e]], { padding: [20, 20] });
}

// ------------------------------------------------------------------ form setup
const today = new Date();
$("date").value = today.toISOString().slice(0, 10);
$("d-date").value = new Date(today.getTime() - 864e5).toISOString().slice(0, 10);
// Time zones as UTC offsets with their standard names (no city list). Daylight-saving zones appear
// under the offset in effect, e.g. CEST (Central European Summer Time) under UTC+02:00.
const ZONES = [
  [-720, "AoE (Anywhere on Earth)"], [-660, "SST (Samoa)"], [-600, "HST (Hawaii)"], [-570, "MART (Marquesas)"],
  [-540, "AKST (Alaska) / HDT"], [-480, "PST (Pacific) / AKDT"], [-420, "MST (Mountain) / PDT"],
  [-360, "CST (Central) / MDT"], [-300, "EST (Eastern) / CDT"], [-240, "AST (Atlantic) / EDT"], [-210, "NST (Newfoundland)"],
  [-180, "BRT (Brasília), ART / ADT"], [-120, "GST (South Georgia)"], [-60, "AZOT (Azores), CVT"],
  [0, "UTC / GMT, WET"], [60, "CET (Central European), WAT / BST, WEST"], [120, "EET (Eastern European), CAT, SAST / CEST"],
  [180, "MSK (Moscow), EAT, AST (Arabia) / EEST"], [210, "IRST (Iran)"], [240, "GST (Gulf)"], [270, "AFT (Afghanistan)"],
  [300, "PKT (Pakistan)"], [330, "IST (India)"], [345, "NPT (Nepal)"], [360, "BST (Bangladesh)"], [390, "MMT (Myanmar)"],
  [420, "ICT (Indochina), WIB"], [480, "CST (China), SGT, AWST"], [540, "JST (Japan), KST (Korea)"],
  [570, "ACST (Central Australia)"], [600, "AEST (Eastern Australia)"], [630, "LHST (Lord Howe)"], [660, "SBT, AEDT"],
  [720, "NZST (New Zealand)"], [765, "CHAST (Chatham)"], [780, "NZDT, TOT (Tonga)"], [840, "LINT (Line Islands)"],
];
const offLabel = (m) => m === 0 ? "UTC" : `UTC${m > 0 ? "+" : "-"}${String(Math.floor(Math.abs(m) / 60)).padStart(2, "0")}:${String(Math.abs(m) % 60).padStart(2, "0")}`;
const localOffset = -new Date().getTimezoneOffset();  // minutes east of UTC, DST included
$("tz").innerHTML = ZONES.map(([m, name]) => `<option value="${offLabel(m)}" ${m === localOffset ? "selected" : ""}>${esc(offLabel(m) === "UTC" ? "UTC±00:00" : offLabel(m))} · ${esc(name)}</option>`).join("");
const browserTz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

fetch("/api/config").then((r) => r.json()).then((cfg) => {
  $("continent").innerHTML = cfg.continents.map((c) => `<option>${esc(c)}</option>`).join("");
  const guess = browserTz.startsWith("Europe") ? "Europe" : browserTz.startsWith("America") ? "Continental US" : cfg.continents[0];
  $("continent").value = guess;
  updateRegion();
});

document.querySelectorAll("#where-mode button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#where-mode button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  state.whereMode = b.dataset.v;
  document.querySelectorAll(".where-panel").forEach((p) => { p.hidden = p.dataset.for !== state.whereMode; });
  updateRegion();
}));
["continent", "pad", "bb-s", "bb-n", "bb-w", "bb-e"].forEach((id) => $(id).addEventListener("change", updateRegion));

$("geocode").addEventListener("click", async () => {
  const q = $("place").value.trim();
  if (!q) return;
  $("place-result").textContent = "Looking up…";
  const r = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
  if (!r.ok) { $("place-result").textContent = (await r.json()).detail || "Not found."; state.placeBox = null; return; }
  const hit = await r.json();
  state.placeBox = { bbox: hit.bbox, label: hit.name };
  $("place-result").textContent = `Found: ${hit.name}`;
  updateRegion();
});
$("place").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("geocode").click(); } });

function whereSpec() {
  if (state.whereMode === "continent") return { mode: "continent", continent: $("continent").value };
  if (state.whereMode === "place") return state.placeBox ? { mode: "place", bbox: state.placeBox.bbox, label: state.placeBox.label, pad_km: +$("pad").value } : null;
  const v = ["bb-s", "bb-n", "bb-w", "bb-e"].map((id) => parseFloat($(id).value));
  return v.every(Number.isFinite) ? { mode: "bbox", bbox: v, label: "custom box" } : null;
}

async function updateRegion() {
  const w = whereSpec();
  if (!w) { drawRegion(null); state.region = null; return; }
  const r = await fetch(`/api/region?where=${encodeURIComponent(JSON.stringify(w))}`);
  if (r.ok) { state.region = await r.json(); drawRegion(state.region); }
}

function buildMeta() {
  const unc = $("unc").value;
  const known = [parseFloat($("known-lat").value), parseFloat($("known-lon").value)];
  return {
    where: whereSpec(),
    when: { date: $("date").value, time: unc === "day" ? null : ($("time").value || null), tz: $("tz").value, uncertainty_min: unc === "day" ? null : +unc },
    temp_c: $("temp").value ? +$("temp").value : null,
    environment: $("env").value, height_m: $("height").value ? +$("height").value : null, notes: $("notes").value,
    known_location: known.every(Number.isFinite) ? { lat: known[0], lon: known[1] } : null,
  };
}

// ------------------------------------------------------------------ upload
async function uploadFile(file) {
  $("drop-main").textContent = `Uploading ${file.name}…`;
  $("btn-analyze").disabled = true;
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) { $("drop-main").textContent = `Upload failed: ${(await r.json()).detail}`; return; }
  const u = await r.json();
  state.uploadId = u.upload_id;
  $("drop-main").textContent = `${file.name} (${(u.bytes / 1e6).toFixed(1)} MB) ready`;
  $("btn-analyze").disabled = false;
  $("s1-status").textContent = "Fill in what you know, then check the audio.";
}
$("file").addEventListener("change", (e) => e.target.files[0] && uploadFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]));

// ------------------------------------------------------------------ jobs
function poll(jobId, progressEl, onDone, onError) {
  progressEl.hidden = false;
  const bar = progressEl.querySelector(".bar"), msg = progressEl.querySelector(".msg");
  const tick = async () => {
    const r = await fetch(`/api/jobs/${jobId}`);
    const j = await r.json();
    bar.style.width = `${Math.round(100 * j.progress)}%`;
    msg.textContent = `${Math.round(100 * j.progress)}%  ${j.message || j.status}  ·  ${fmtDur(j.elapsed_s)}`;
    if (j.status === "done") onDone(j.result);
    else if (j.status === "error") { msg.textContent = j.error; onError && onError(j); }
    else setTimeout(tick, 1000);
  };
  tick();
}
const fmtDur = (s) => (s == null ? "" : s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

// ------------------------------------------------------------------ step 2: analysis
$("btn-analyze").addEventListener("click", async () => {
  const meta = buildMeta();
  if (!meta.where) { $("s1-status").textContent = "Set where it was recorded (at least a continent)."; return; }
  if (!meta.when.date) { $("s1-status").textContent = "Set the date."; return; }
  $("step2").setAttribute("aria-disabled", "false");
  $("analysis").hidden = true; $("btn-search").disabled = true;
  const fd = new FormData(); fd.append("upload_id", state.uploadId); fd.append("meta", JSON.stringify(meta));
  const j = await (await fetch("/api/analyze", { method: "POST", body: fd })).json();
  poll(j.id, $("p2"), renderAnalysis);
});

function renderAnalysis(a) {
  state.analysis = a;
  $("analysis").hidden = false;
  const v = a.verdict;
  $("grade").className = `pill ${v.grade}`; $("grade").textContent = v.grade;
  $("grade-text").textContent = v.searchable ? "This recording can be searched." : "This recording cannot be searched as it is.";
  $("reasons").innerHTML = v.reasons.map((r) => `<li>${esc(r)}</li>`).join("");
  $("tips").innerHTML = v.tips.map((r) => `<li>${esc(r)}</li>`).join("");
  const hum = a.hum.grid_hz ? `${a.hum.grid_hz} Hz` : "not detected";
  $("kpis").innerHTML = [
    ["Length", fmtDur(a.duration_s)], ["Aircraft curves", `${v.n_air} (${v.n_tier_a} strong)`], ["Loud passes", v.n_loud],
    ["Road vehicles", v.n_ground], ["Mains hum", hum], ["Noise bursts", `${Math.round(100 * a.noise.transient_fraction)}% of time`],
  ].map(([k, val]) => `<div class="kpi"><span>${k}</span><b>${esc(val)}</b></div>`).join("") +
    (a.hum.hint ? `<p class="hint" style="grid-column:1/-1">Mains hum at ${a.hum.grid_hz} Hz is typical of: ${esc(a.hum.hint)}.</p>` : "");
  $("spec").src = `data:image/jpeg;base64,${a.spectrogram_jpg}`;
  const rows = a.events.map((e, i) => `<tr class="${e.tier === "G" ? "road" : ""}"><td>${i + 1}</td><td>${e.tier === "G" ? "road vehicle" : e.tier === "A" ? "aircraft (strong)" : "aircraft (weak)"}</td><td>${fmtClock(e.t)}</td><td>${e.v} ± ${e.v_sigma} m/s</td><td>${(e.r / 1000).toFixed(2)} ± ${(e.r_sigma / 1000).toFixed(2)} km</td><td>${e.f0} Hz</td><td>${e.ridges}</td></tr>`).join("");
  $("events").innerHTML = a.events.length ? `<tr><th>#</th><th>kind</th><th>closest at</th><th>speed</th><th>closest distance</th><th>tone</th><th>harmonics</th></tr>${rows}` : "";
  renderData(a.data);
  if (a.estimate) {
    const e = a.estimate;
    $("estimate").innerHTML = `Estimated time: <b>${fmtDur(e.low)} – ${fmtDur(e.high)}</b> <span class="hint">(search ${fmtDur(e.breakdown.search)}, candidate checks ${fmtDur(e.breakdown.consistency)}, refinement ${fmtDur(e.breakdown.refine)}${e.breakdown.data ? `, flight data ${fmtDur(e.breakdown.data)}` : ""})</span>`;
  } else $("estimate").textContent = "";
  updateSearchButton();
}
const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

function renderData(d) {
  const el = $("data-line");
  if (!d || d.error) { el.innerHTML = `Flight data: ${esc(d?.error || "unknown")}`; return; }
  const items = Object.entries(d.days).map(([day, s]) => `<li>${esc(day)}: ${esc(s.text)}</li>`).join("");
  const needs = Object.entries(d.days).filter(([, s]) => s.state === "remote" || s.state === "raw").map(([day]) => day);
  el.innerHTML = `<b>Flight data</b> for ${esc(d.region.label)} (${esc(d.region.adsb_store)}):<ul class="reasons">${items}</ul>` +
    (needs.length ? `<button id="prep-btn">Download &amp; prepare (${needs.length} day${needs.length > 1 ? "s" : ""})</button><div class="progress" id="pprep" hidden><div class="bar"></div><span class="msg"></span></div>` : "");
  if (needs.length) $("prep-btn").addEventListener("click", () => prepareDays(needs, d.region.adsb_store));
}

async function prepareDays(days, store) {
  $("prep-btn").disabled = true;
  for (const day of days) {
    const fd = new FormData(); fd.append("date", day); fd.append("store", store);
    const j = await (await fetch("/api/data/prepare", { method: "POST", body: fd })).json();
    await new Promise((res) => poll(j.id, $("pprep"), res, res));
  }
  for (const day of days) state.analysis.data.days[day] = await (await fetch(`/api/data/status?date=${day}&store=${store}`)).json();
  state.analysis.data.state = Object.values(state.analysis.data.days).every((s) => s.state === "ready") ? "ready" : "missing";
  renderData(state.analysis.data); updateSearchButton();
}

function updateSearchButton() {
  const a = state.analysis;
  const ready = a && a.data && a.data.state === "ready";
  $("btn-search").disabled = !(a && a.verdict.searchable && ready);
  $("s2-status").textContent = !a ? "" : !a.verdict.searchable ? "Not enough aircraft evidence to search." : !ready ? "Prepare the flight data first." : "Ready.";
}

// ------------------------------------------------------------------ step 3: search
$("btn-search").addEventListener("click", async () => {
  $("step3").setAttribute("aria-disabled", "false"); $("btn-search").disabled = true;
  layers.results.clearLayers();
  $("result-body").innerHTML = `<p class="hint">Searching…</p>`;
  const fd = new FormData(); fd.append("upload_id", state.uploadId); fd.append("meta", JSON.stringify(buildMeta()));
  const j = await (await fetch("/api/search", { method: "POST", body: fd })).json();
  poll(j.id, $("p3"), renderResults, () => { $("btn-search").disabled = false; });
});

function renderResults(res) {
  $("btn-search").disabled = false;
  layers.results.clearLayers();
  if (!res.best) { $("result-body").innerHTML = `<p>${esc(res.message || "No result.")}</p>`; return; }
  res.candidates.slice(1).forEach((c) => L.circleMarker([c.lat, c.lon], { radius: 6, color: "#ffb547", fillColor: "#ffb547", fillOpacity: .8, weight: 1 })
    .bindTooltip(`#${c.rank} · ${esc(c.flight_a)} + ${esc(c.flight_b)}`).addTo(layers.results));
  res.tracks.forEach((t) => L.polyline(t.points, { color: "#5fd4e6", weight: 2, opacity: .85 }).bindTooltip(`aircraft ${esc(t.icao)}`).addTo(layers.results));
  const b = res.best;
  L.circleMarker([b.lat, b.lon], { radius: 11, color: "#06d6a0", fillColor: "#06d6a0", fillOpacity: .9, weight: 2 }).bindTooltip("best location").addTo(layers.results);
  const known = buildMeta().known_location;
  if (known) L.circleMarker([known.lat, known.lon], { radius: 8, color: "#ff6b7d", weight: 3, fill: false }).bindTooltip("known location (testing)").addTo(layers.results);
  map.setView([b.lat, b.lon], 11);
  const confText = res.confidence || "n/a";
  const ev = res.evaluation ? `<div class="kpi"><span>Error vs known location</span><b>${res.evaluation.best_error_m} m</b></div><div class="kpi"><span>Rank of a hit within 1.5 km</span><b>${res.evaluation.rank_within_1500m ?? "not found"}</b></div>` : "";
  const rows = res.candidates.slice(0, 12).map((c) => `<tr><td>${c.rank}</td><td>${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}</td><td>${esc(c.t0_utc)}</td><td>${esc(c.flight_a)}</td><td>${esc(c.flight_b)}</td><td>${(c.combined ?? 0).toFixed(2)}</td><td>${(c.loud ?? 0).toFixed(2)}</td><td>${(c.spectro ?? 0).toFixed(1)}</td></tr>`).join("");
  $("result-body").innerHTML = `
    <div class="best-card">
      <div class="kpi"><span>Best location</span><b>${b.lat.toFixed(5)}, ${b.lon.toFixed(5)}</b></div>
      <div class="kpi"><span>Recording started (UTC)</span><b>${esc(b.t0_utc)}</b></div>
      <div class="kpi"><span>Matched aircraft</span><b>${esc(b.flight_a)}${b.flight_b ? " + " + esc(b.flight_b) : ""}</b></div>
      <div class="kpi"><span>Confidence</span><b>${esc(confText)}</b></div>${ev}
    </div>
    <p class="hint"><a href="https://www.google.com/maps/search/?api=1&query=${b.lat},${b.lon}" target="_blank" rel="noopener">Open in Google Maps</a> · ${res.candidates.length} candidates kept from ${esc(res.stats.n_events)} events · search took ${fmtDur(res.stats.seconds)}</p>
    <div class="table-wrap"><table><tr><th>#</th><th>location</th><th>start (UTC)</th><th>aircraft A</th><th>aircraft B</th><th>overall</th><th>loudness fit</th><th>tone fit</th></tr>${rows}</table></div>
    ${res.overlays.map((o) => `<img class="overlay" alt="Predicted curve at the best location versus the recording" src="data:image/jpeg;base64,${o}">`).join("")}`;
}

// ------------------------------------------------------------------ data tab
async function loadIngested() {
  const cfg = await (await fetch("/api/config")).json();
  $("ingested").innerHTML = cfg.ingested.length
    ? `<tr><th>date</th><th>region</th><th>aircraft</th><th>positions</th></tr>` + cfg.ingested.map((d) => `<tr><td>${esc(d.date)}</td><td>${esc(d.region)}</td><td>${d.aircraft?.toLocaleString() ?? "?"}</td><td>${d.points ? (d.points / 1e6).toFixed(1) + " M" : "?"}</td></tr>`).join("")
    : `<tr><td>No flight data prepared yet.</td></tr>`;
}
$("d-check").addEventListener("click", async () => {
  const s = await (await fetch(`/api/data/status?date=${$("d-date").value}&store=${$("d-store").value}`)).json();
  $("d-status").textContent = s.text;
});
$("d-prepare").addEventListener("click", async () => {
  const fd = new FormData(); fd.append("date", $("d-date").value); fd.append("store", $("d-store").value);
  const r = await fetch("/api/data/prepare", { method: "POST", body: fd });
  if (!r.ok) { $("d-status").textContent = (await r.json()).detail; return; }
  poll((await r.json()).id, $("pd"), () => { $("d-status").textContent = "Done."; loadIngested(); });
});
