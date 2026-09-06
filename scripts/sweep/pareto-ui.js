import {frontier, memoryValue, defaultState, readState, writeState, measurementsCsv} from "./pareto.mjs";
import {fmt, fmtSignificant} from "./charts.js";
import {fetchJson} from "./decode.js";
import {createPlots, formatSize, workloadName} from "./pareto-plots.js";

const $ = id => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};
let datasetIndex, measurements = [], experiments = new Map(), workloads = new Map();
let state, frontierResult, sortedRows, tableButtons = [], plots, blockSizes = [];

function remember(replace = false) {
  const url = `${location.pathname}?${writeState(state)}${location.hash}`;
  if (url !== location.pathname + location.search + location.hash) history[replace ? "replaceState" : "pushState"](null, "", url);
}

function restore() {
  state = readState(location.search, [...experiments.keys()], [...workloads.keys()], blockSizes);
  if (state.selected && !measurements.some(row => row.id === state.selected)) state.selected = null;
  if (state.layout === "overlay" && state.workload === "all") state.workload = workloads.keys().next().value;
}

function syncControls() {
  for (const checkbox of $("systems").querySelectorAll("input")) checkbox.checked = state.systems.includes(checkbox.value);
  for (const key of ["codecs", "shuffles", "blocks"]) {
    for (const option of $(key).options) option.selected = state[key] == null ? option.value === "all"
      : state[key].includes(key === "blocks" ? +option.value : option.value);
  }
  for (const key of ["mode", "view", "layout", "workload"]) $(key).value = state[key];
  $("budget").value = state.budget ?? "";
  $("workload").options[0].disabled = state.layout === "overlay";
  $("fit").setAttribute("aria-pressed", String(state.extent === "fit"));
}

function change() {
  if (state.layout === "overlay" && state.workload === "all") state.workload = workloads.keys().next().value;
  syncControls();
  remember();
  render();
}

function wireControls() {
  $("settings").open = !matchMedia("(max-width: 760px)").matches;
  $("systems").replaceChildren();
  for (const e of experiments.values()) {
    const label = el("label", "system-choice"), check = el("input");
    check.type = "checkbox"; check.value = e.id;
    const caption = el("span");
    caption.append(el("strong", null, e.label), el("small", null, `${e.start_utc.slice(0, 10)} UTC · ${e.repetitions} repetitions${e.summary_only ? " · summary only" : ""}`));
    label.append(check, caption); $("systems").append(label);
    check.addEventListener("change", () => { state.systems = [...$("systems").querySelectorAll("input:checked")].map(n => n.value); change(); });
  }
  for (const block of blockSizes) $("blocks").append(new Option(formatSize(block), block));
  for (const w of workloads.values()) $("workload").append(new Option(workloadName(w), w.id));
  for (const key of ["codecs", "shuffles", "blocks"]) {
    $(key).addEventListener("change", event => {
      const selected = [...event.target.selectedOptions].map(o => o.value);
      // Choosing All clears other selections; choosing a value from All starts a subset.
      const previous = state[key];
      state[key] = selected.includes("all") && previous != null ? null
        : selected.filter(v => v !== "all").map(v => key === "blocks" ? +v : v);
      if (selected.length === 1 && selected[0] === "all") state[key] = null;
      change();
    });
  }
  for (const key of ["mode", "view", "layout", "workload"]) $(key).addEventListener("change", () => { state[key] = $(key).value; change(); });
  $("budget").addEventListener("change", () => {
    if (!$("budget").checkValidity()) { $("budget").reportValidity(); return; }
    state.budget = $("budget").value === "" ? null : +$("budget").value; change();
  });
  $("filters").addEventListener("submit", event => event.preventDefault());
  $("reset-filters").addEventListener("click", () => { state = defaultState([...experiments.keys()]); change(); });
  $("fit").addEventListener("click", () => { state.extent = "fit"; change(); });
  $("full").addEventListener("click", () => { state.extent = "full"; change(); });
  $("download").addEventListener("click", () => {
    const csv = measurementsCsv(sortedRows, frontierResult.ids, experiments);
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const a = el("a"); a.href = url; a.download = "blosc-filtered-measurements.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("popstate", () => { restore(); syncControls(); render(); });
}

function render() {
  frontierResult = frontier(measurements, state);
  $("count").textContent = `${frontierResult.candidates.length} / ${measurements.length} measurements · ${frontierResult.ids.size} frontier settings`;
  $("empty").hidden = frontierResult.candidates.length !== 0;
  $("download").disabled = !frontierResult.candidates.length;
  $("chart-note").textContent = "Prominent points maximize throughput and fold. " +
    `Frontiers stay within each system${state.mode === "codec" ? " and codec" : ""}. ` +
    `${state.extent === "fit" ? "Zoomed to candidates" : "Full extent"}; logarithmic horizontal axis. ` +
    "Scales match within each row. Focus a chart and use arrow keys to see repetition min–max.";
  plots.render(state, frontierResult);
  renderTable();
  renderDetail();
}

const columns = [
  ["system", "System", r => experiments.get(r.experiment_id).label],
  ["workload", "Input / chunk", r => workloadName(workloads.get(r.workload_id))],
  ["codec", "Codec / setting", r => r.codec], ["shuffle", "Shuffle", r => r.shuffle],
  ["block", "Block", r => r.block_kib, r => formatSize(r.block_kib)],
  ["throughput", "Input GiB/s", r => r.throughput_gibs.median, r => fmt(r.throughput_gibs.median)],
  ["fold", "Fold ×", r => r.compression_fold, r => fmt(r.compression_fold)],
  ["measured", "Measured GiB", r => r.measured_device_gib?.median, r => fmt(r.measured_device_gib?.median)],
  ["estimated", "Estimated GiB", r => memoryValue(r), r => fmt(memoryValue(r))],
  ["repetitions", "Repetitions", r => r.repetitions],
];

function renderTable() {
  const getter = columns.find(c => c[0] === state.sort)[2];
  sortedRows = [...frontierResult.candidates].sort((a, b) => {
    const av = getter(a), bv = getter(b);
    if (av == null || bv == null) return av == null && bv == null ? a.id.localeCompare(b.id, "en") : av == null ? 1 : -1;
    const diff = typeof av === "number" ? av - bv : av.localeCompare(bv, "en");
    return (state.direction === "asc" ? diff : -diff) || a.id.localeCompare(b.id, "en");
  });
  $("table-head").replaceChildren();
  for (const [key, label] of columns) {
    const th = el("th"), button = el("button", null, label + (state.sort === key ? state.direction === "asc" ? " ↑" : " ↓" : ""));
    th.scope = "col"; th.setAttribute("aria-sort", state.sort === key ? state.direction === "asc" ? "ascending" : "descending" : "none");
    button.type = "button";
    button.addEventListener("click", () => {
      state.direction = state.sort === key && state.direction === "desc" ? "asc" : "desc"; state.sort = key;
      remember(); renderTable(); $("table-head").children[columns.findIndex(c => c[0] === key)].querySelector("button").focus();
    });
    th.append(button); $("table-head").append(th);
  }
  const frontierHeader = el("th", null, "Membership"); frontierHeader.scope = "col"; $("table-head").append(frontierHeader);
  const fragment = document.createDocumentFragment(); tableButtons = [];
  for (const [i, row] of sortedRows.entries()) {
    const tr = el("tr"); tr.dataset.id = row.id;
    for (const [key, , get, display] of columns) {
      const td = el("td");
      if (key === "codec") {
        const button = el("button", "setting-button", row.codec);
        button.type = "button"; button.setAttribute("aria-label", `Inspect ${experiments.get(row.experiment_id).label}, ${workloadName(workloads.get(row.workload_id))}, ${row.codec}, ${row.shuffle}, ${formatSize(row.block_kib)}`);
        button.addEventListener("click", () => select(row.id));
        button.addEventListener("focus", () => select(row.id, false));
        button.addEventListener("keydown", event => {
          let next;
          if (event.key === "ArrowDown") next = Math.min(i + 1, tableButtons.length - 1);
          else if (event.key === "ArrowUp") next = Math.max(i - 1, 0);
          else if (event.key === "Home") next = 0;
          else if (event.key === "End") next = tableButtons.length - 1;
          else return;
          event.preventDefault(); tableButtons[next].focus();
        });
        td.append(button); tableButtons.push(button);
      } else td.textContent = (display ?? get)(row);
      tr.append(td);
    }
    tr.append(el("td", frontierResult.ids.has(row.id) ? "frontier-label" : null, row.control ? "Raw control" : frontierResult.ids.has(row.id) ? "Frontier" : "Candidate"));
    fragment.append(tr);
  }
  $("table-body").replaceChildren(fragment); highlightTable();
}

function highlightTable() {
  let active = sortedRows.findIndex(r => r.id === state.selected);
  if (active < 0) active = 0;
  for (const [i, tr] of [...$("table-body").children].entries()) {
    const selected = tr.dataset.id === state.selected;
    tr.classList.toggle("selected", selected);
    tableButtons[i].setAttribute("aria-pressed", String(selected)); tableButtons[i].tabIndex = i === active ? 0 : -1;
  }
}

function select(id, push = true) {
  if (id === state.selected) return;
  state.selected = id; remember(!push);
  plots.updateSelection(); highlightTable(); renderDetail();
  // Move the table's own scroll viewport, without jumping the page away from a chart.
  const tr = [...$("table-body").children].find(r => r.dataset.id === id), scroll = document.querySelector(".table-scroll");
  if (tr) {
    const top = tr.getBoundingClientRect().top - scroll.getBoundingClientRect().top + scroll.scrollTop;
    if (top < scroll.scrollTop + 45 || top + tr.offsetHeight > scroll.scrollTop + scroll.clientHeight)
      scroll.scrollTop = top - 50;
  }
}

function link(href, label) { const a = el("a", null, label); a.href = href; return a; }

function renderDetail() {
  const panel = $("detail-content"), row = measurements.find(r => r.id === state.selected);
  document.body.classList.toggle("has-selection", Boolean(row));
  panel.replaceChildren();
  if (!row) { panel.textContent = "Select a point or table setting to see measurements and provenance."; return; }
  const e = experiments.get(row.experiment_id), w = workloads.get(row.workload_id);
  panel.append(el("h3", null, `${e.label} · ${row.codec}`), el("p", null, `${workloadName(w)} · ${row.shuffle} shuffle · ${formatSize(row.block_kib)} block · level ${row.level}`));
  const visible = frontierResult.candidates.some(candidate => candidate.id === row.id);
  panel.append(el("p", null, !visible ? "This selection is outside the current filters." : row.control ? "Raw control; excluded from frontier membership." : frontierResult.ids.has(row.id) ? "On the current frontier." : "Dominated or missing an objective for the current frontier."));
  const dl = el("dl");
  for (const [label, value] of [
    ["Median input throughput", `${fmtSignificant(row.throughput_gibs.median)} GiB/s`],
    [`Min–max (${row.repetitions} measured repetitions)`, `${fmtSignificant(row.throughput_gibs.min)}–${fmtSignificant(row.throughput_gibs.max)} GiB/s`],
    ["Reported compression fold", `${fmtSignificant(row.compression_fold)}×`],
    ["Measured device delta", row.measured_device_gib.median == null ? "Unavailable" : `${fmtSignificant(row.measured_device_gib.median)} GiB`],
    ["Measured memory min–max", row.measured_device_gib.min == null ? "Unavailable in this archive" : `${fmtSignificant(row.measured_device_gib.min)}–${fmtSignificant(row.measured_device_gib.max)} GiB`],
    ["Estimated device allocation", row.estimated_device_gib == null ? "Unavailable" : `${fmtSignificant(row.estimated_device_gib)} GiB`],
    ["Estimated pinned host allocation", row.estimated_pinned_gib == null ? "Unavailable" : `${fmtSignificant(row.estimated_pinned_gib)} GiB`],
    ["Acquired (UTC)", `${e.start_utc} to ${e.finish_utc}`], ["Source revision", e.source_commit],
  ]) { const group = el("div"); group.append(el("dt", null, label), el("dd", null, value)); dl.append(group); }
  panel.append(dl, el("p", null, e.summary_only ? "Summary only: the original repetitions are not retained. Min–max values come from the preserved summary." : `${e.warmups} warmup per setting excluded. Min–max is observed spread, not a confidence interval.`));
  const links = el("ul");
  for (const [href, label] of [[row.provenance.summary, `Original summary (line ${row.provenance.summary_line})`], [row.provenance.metadata, "Original provenance"], [row.provenance.raw, "Raw repetitions (.jsonl.gz)"]]) {
    if (href) { const li = el("li"); li.append(link(href, label)); links.append(li); }
  }
  panel.append(links);
  const references = row.samples?.map(({repeat, raw_line}) => ({repeat, raw_line})) ?? null;
  const more = el("details"); more.append(el("summary", null, "Record references"),
    el("pre", null, JSON.stringify({configuration_id: row.configuration_id, samples: references}, null, 2)));
  panel.append(more);
}

function renderMethodology() {
  const names = {throughput_gibs: "Input throughput (GiB/s)", compression_fold: "Reported compression fold (×)",
    measured_device_gib: "Measured device memory", estimated_device_gib: "Estimated device allocation", estimated_pinned_gib: "Pinned host allocation", ranges: "Repetition ranges", comparison: "Comparing systems"};
  $("definitions").replaceChildren(); $("experiments").replaceChildren();
  const manifest = el("p"); manifest.append(link("data/pareto/manifest.json", "Dataset manifest"));
  $("experiments").append(manifest);
  for (const [key, text] of Object.entries(datasetIndex.definitions)) {
    const p = el("p"); p.append(el("strong", null, names[key] ?? key), document.createTextNode(text)); $("definitions").append(p);
  }
  for (const e of experiments.values()) {
    const details = el("details"); details.append(el("summary", null, `${e.label} · ${e.start_utc.slice(0, 10)} UTC${e.summary_only ? " · summary only" : ""}`));
    details.append(el("p", null, e.methodology), el("p", null, Object.entries(e.hardware).map(([k,v]) => `${k}: ${v ?? "not recorded"}`).join("; ")),
      el("p", null, `Source: ${e.source_commit}. ${e.configuration_count} configurations. ${e.validated_executions ?? "No retained raw"} executions validated.`));
    for (const note of e.notes) details.append(el("p", null, note));
    const files = el("div", "artifact-links"); for (const f of e.files) files.append(link(f.href, f.label));
    details.append(files); $("experiments").append(details);
  }
}

function renderScenarios() {
  const scenarios = new Map();
  for (const workload of workloads.values()) scenarios.set(workload.scenario, workload.source_url);
  const note = $("scenario-note"); note.replaceChildren();
  note.append(document.createTextNode(`Measurements were taken using ${scenarios.size === 1 ? "the " : ""}`));
  [...scenarios].forEach(([scenario, source], i) => {
    if (i) note.append(document.createTextNode(i === scenarios.size - 1 ? " and " : ", "));
    if (source) note.append(link(source, `${scenario} scenario`));
    else note.append(el("code", null, scenario));
  });
  note.append(document.createTextNode(scenarios.size === 1 ? "." : " scenarios."));
}

async function boot() {
  $("retry").hidden = true; $("load-status").hidden = false; $("load-status").textContent = "Loading retained measurements…";
  try {
    if (typeof d3 === "undefined") throw new Error("The local D3 bundle is missing. Rebuild the report site.");
    datasetIndex = await fetchJson("data/pareto/index.json");
    if (datasetIndex.version !== 1 || !Array.isArray(datasetIndex.experiments)) throw new Error("Unsupported Pareto dataset index. Rebuild the report site.");
    const data = await Promise.all(datasetIndex.experiments.map(experiment => fetchJson(experiment.data)));
    for (const [i, d] of data.entries()) {
      if (d.version !== 1 || d.experiment.id !== datasetIndex.experiments[i].id || !Array.isArray(d.measurements) || !Array.isArray(d.workloads)) throw new Error("Invalid experiment data. Rebuild the report site.");
    }
    measurements = data.flatMap(d => d.measurements);
    if (new Set(measurements.map(row => row.id)).size !== measurements.length) throw new Error("Duplicate measurement identities. Rebuild the report site.");
    experiments = new Map(datasetIndex.experiments.map(experiment => [experiment.id, experiment]));
    const unique = new Map(data.flatMap(d => d.workloads).map(w => [w.id, w]));
    workloads = new Map([...unique].sort((a,b) => b[1].fill.localeCompare(a[1].fill, "en") || a[1].chunk_kib - b[1].chunk_kib || a[0].localeCompare(b[0], "en")));
    blockSizes = [...new Set(measurements.map(row => row.block_kib))].sort((a,b) => a-b);
    plots = createPlots({container: $("charts"), legend: $("legend"), experiments, workloads, onSelect: select});
    restore(); wireControls(); syncControls(); renderScenarios(); renderMethodology();
    $("workspace").hidden = false; $("load-status").hidden = true; render();
    if (!measurements.length) { $("empty").hidden = false; $("empty").textContent = "No retained datasets are listed. Add an experiment to the dataset manifest and rebuild the site."; }
  } catch (error) {
    $("workspace").hidden = true; $("load-status").textContent = `Unable to load Blosc measurements: ${error.message}. Serve the generated site over HTTP and check the data files.`;
    $("load-status").setAttribute("role", "alert"); $("retry").hidden = false;
  }
}

wireThemeToggle(() => plots?.redraw());
let lastWidth = 0;
const resize = new ResizeObserver(entries => {
  const width = Math.round(entries[0].contentRect.width);
  if (plots && width > 0 && width !== lastWidth) { lastWidth = width; plots.redraw(); }
});
resize.observe($("charts"));
$("retry").addEventListener("click", boot);
boot();
