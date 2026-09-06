import {frontier, memoryValue, defaultState, readState, writeState, measurementsCsv} from "./pareto.mjs";
import {plotAxes, fmt, fmtSignificant} from "./charts.js";
import {fetchJson} from "./decode.js";

const $ = id => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};
const size = kib => kib === 0 ? "Raw" : kib >= 1024 ? `${kib / 1024} MiB` : `${kib} KiB`;
const inputName = fill => fill === "rand" ? "12-bit random" : fill === "xor" ? "Coordinate XOR" : fill;
const workloadName = w => `${inputName(w.fill)} / ${size(w.chunk_kib)} chunks`;
const shape = row => row.control ? d3.symbolDiamond : ({none: d3.symbolSquare, byte: d3.symbolCircle, bit: d3.symbolTriangle}[row.shuffle] ?? d3.symbolCross);
const color = row => row.codec.endsWith("lz4") ? "var(--codec-lz4)" : "var(--codec-zstd)";
const dash = index => [null, "7 4", "2 4", "8 3 2 3"][index % 4];
let index, rows = [], experiments = new Map(), workloads = new Map(), state, result, sortedRows;
let blocks = [], chartRedraws = [], tableButtons = [];

function systemFill(svg, id, systemIndex, ink) {
  if (systemIndex === 0) return ink;
  if (systemIndex === 1) return "var(--surface-1)";
  // Filled, open, then hatched marks keep systems identifiable in an overlay.
  // Shape still means shuffle.
  let defs = svg.select("defs");
  if (defs.empty()) defs = svg.append("defs");
  if (defs.select(`#${id}`).empty()) {
    const pattern = defs.append("pattern").attr("id", id).attr("patternUnits", "userSpaceOnUse")
      .attr("width", 4).attr("height", 4).attr("patternTransform", `rotate(${45 + (systemIndex - 2) * 30})`);
    pattern.append("rect").attr("width", 4).attr("height", 4).attr("fill", "var(--surface-1)");
    pattern.append("path").attr("d", "M0,0V4").attr("stroke", ink).attr("stroke-width", 2);
  }
  return `url(#${id})`;
}

function remember(replace = false) {
  const url = `${location.pathname}?${writeState(state)}${location.hash}`;
  if (url !== location.pathname + location.search + location.hash) history[replace ? "replaceState" : "pushState"](null, "", url);
}

function restore() {
  state = readState(location.search, [...experiments.keys()], [...workloads.keys()], blocks);
  if (state.selected && !rows.some(r => r.id === state.selected)) state.selected = null;
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
  for (const block of blocks) $("blocks").append(new Option(size(block), block));
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
    const csv = measurementsCsv(sortedRows, result.ids, experiments);
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const a = el("a"); a.href = url; a.download = "blosc-filtered-measurements.csv"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  window.addEventListener("popstate", () => { restore(); syncControls(); render(); });
}

function legendMark(label, row, {faint = false, line = false, lineIndex = 0} = {}) {
  const node = el("span", "legend-item");
  const svg = d3.create("svg").attr("viewBox", "-11 -11 22 22").attr("aria-hidden", "true");
  if (line) {
    svg.append("path").attr("d", "M-11,0H11").attr("stroke", "var(--text-secondary)")
      .attr("stroke-width", 2).attr("stroke-dasharray", dash(lineIndex));
    svg.append("circle").attr("r", 5).attr("stroke", "var(--text-secondary)").attr("stroke-width", 1.5)
      .attr("fill", systemFill(svg, `legend-system-${lineIndex}`, lineIndex, "var(--text-secondary)"));
  }
  else svg.append("path").attr("d", d3.symbol().type(shape(row)).size(65)())
    .attr("fill", row.control ? "var(--surface-1)" : row.codec ? color(row) : "var(--text-secondary)")
    .attr("stroke", row.codec ? color(row) : "var(--text-secondary)").attr("opacity", faint ? .3 : 1);
  node.append(svg.node(), document.createTextNode(label));
  return node;
}

function renderLegend() {
  $("legend").replaceChildren(
    legendMark("LZ4", {codec: "lz4", shuffle: "byte"}), legendMark("Zstd", {codec: "zstd", shuffle: "byte"}),
    legendMark("No shuffle", {shuffle: "none"}), legendMark("Byte shuffle", {shuffle: "byte"}),
    legendMark("Bitshuffle", {shuffle: "bit"}), legendMark("Other candidates", {shuffle: "byte"}, {faint: true}));
  $("legend").append(legendMark("Raw control (excluded)", {control: true}));
  if (state.layout === "overlay") state.systems.forEach((id, i) => $("legend").append(legendMark(experiments.get(id).label, {}, {line: true, lineIndex: i})));
}

function render() {
  result = frontier(rows, state);
  $("count").textContent = `${result.candidates.length} / ${rows.length} measurements · ${result.ids.size} frontier settings`;
  $("empty").hidden = result.candidates.length !== 0;
  $("download").disabled = !result.candidates.length;
  $("chart-note").textContent = "Prominent points maximize throughput and fold. " +
    `Frontiers stay within each system${state.mode === "codec" ? " and codec" : ""}. ` +
    `${state.extent === "fit" ? "Zoomed to candidates" : "Full extent"}; logarithmic horizontal axis. ` +
    "Scales match within each row. Focus a chart and use arrow keys to see repetition min–max.";
  renderLegend();
  renderCharts();
  renderTable();
  renderDetail();
}

function renderCharts() {
  $("charts").replaceChildren(); chartRedraws = [];
  if (!state.systems.length) return;
  for (const w of workloads.values()) {
    if (state.workload !== "all" && w.id !== state.workload) continue;
    const candidates = result.candidates.filter(r => r.workload_id === w.id);
    const plotted = candidates.filter(r => Number.isFinite(xValue(r)) && xValue(r) > 0);
    const section = el("section", "workload-row");
    const title = el("h3", null, workloadName(w));
    title.append(el("small", null, `Shape ${w.shape.join(" × ")} · ${w.padded_batch_bytes / 2**20} MiB batch`));
    section.append(title);
    const matrix = el("div", "matrix-row");
    matrix.tabIndex = 0; matrix.setAttribute("role", "region"); matrix.setAttribute("aria-label", `${workloadName(w)} systems`);
    matrix.style.setProperty("--systems", state.layout === "overlay" ? 1 : state.systems.length);
    section.append(matrix); $("charts").append(section);
    const xs = plotted.map(xValue), ys = plotted.flatMap(r => [r.throughput_gibs.min, r.throughput_gibs.max]);
    const xDomain = logDomain(xs, state.extent === "fit");
    const yDomain = domain(ys, state.extent === "fit");
    const groups = state.layout === "overlay" ? [state.systems] : state.systems.map(id => [id]);
    for (const ids of groups) {
      const cell = el("div", "plot-cell");
      const points = plotted.filter(r => ids.includes(r.experiment_id));
      cell.append(el("h4", null, ids.map(id => experiments.get(id).label).join(" / ")));
      const missing = candidates.filter(r => ids.includes(r.experiment_id)).length - points.length;
      cell.append(el("p", "plot-meta", `${points.length} candidates${missing ? ` · ${missing} unavailable on this axis` : ""}`));
      matrix.append(cell);
      drawChart(cell, points, xDomain, yDomain, w);
    }
  }
}

const xValue = row => state.view === "memory" ? memoryValue(row) : row.compression_fold;

function domain(values, fit) {
  if (!values.length) return [0, 1];
  let lo = d3.min(values), hi = d3.max(values);
  const pad = Math.max((hi - lo) * .06, Math.abs(hi) * .015, .001);
  return [fit ? Math.max(0, lo - pad) : 0, hi + pad];
}

function logDomain(values, fit) {
  if (!values.length) return [1, 10];
  const lo = d3.min(values), hi = d3.max(values);
  // A fold of 1 is the meaningful uncompressed baseline. Keep it for the full
  // view when it is near the data, while fitted views use multiplicative pad.
  const ratio = hi / lo;
  const pad = Math.max(1.025, Math.pow(ratio, .06));
  return [fit ? lo / pad : Math.min(1, lo / pad), hi * pad];
}

function drawChart(cell, points, xDomain, yDomain, workload) {
  const W = Math.max(280, Math.round(cell.getBoundingClientRect().width)), H = 280;
  const margin = {left: 52, top: 14, width: W - 72, height: H - 68};
  const svg = d3.select(cell).append("svg").attr("viewBox", `0 0 ${W} ${H}`)
    .attr("tabindex", points.length ? 0 : -1).attr("role", "group")
    .attr("aria-label", `${cell.querySelector("h4").textContent}, ${workloadName(workload)}. ${points.length} points. Arrow keys select, Home and End jump.`);
  const x = d3.scaleLog().domain(xDomain).range([0, margin.width]);
  const y = d3.scaleLinear().domain(yDomain).range([margin.height, 0]);
  const xLabel = state.view === "compression" ? "Reported compression fold (×)" : "Estimated device allocation (GiB)";
  const g = plotAxes(svg, x, y, {...margin, xLabel, yLabel: "Input throughput (GiB/s)"});
  if (!points.length) {
    g.append("text").attr("class", "plot-empty").attr("x", margin.width / 2).attr("y", margin.height / 2)
      .attr("text-anchor", "middle").text("No eligible measurements");
    return;
  }
  const order = [...points].sort((a, b) => xValue(a) - xValue(b) || b.throughput_gibs.median - a.throughput_gibs.median || a.id.localeCompare(b.id, "en"));
  // Lines connect two-objective frontier settings as a reading aid; they do
  // not interpolate performance between tested settings.
  const groups = d3.group(order.filter(r => result.ids.has(r.id)), r => `${r.experiment_id}:${state.mode === "codec" ? r.codec : "all"}`);
  for (const group of groups.values()) g.append("path").attr("fill", "none")
    .attr("stroke", state.mode === "codec" ? color(group[0]) : "var(--text-secondary)")
    .attr("stroke-width", 1.4).attr("opacity", .65)
    .attr("stroke-dasharray", state.layout === "overlay" ? dash(state.systems.indexOf(group[0].experiment_id)) : null)
    .attr("d", d3.line().x(r => x(xValue(r))).y(r => y(r.throughput_gibs.median))(group));
  const marks = g.append("g").selectAll("g").data([...order].sort((a, b) => Number(result.ids.has(a.id)) - Number(result.ids.has(b.id))))
    .join("g").attr("class", "chart-point").attr("data-id", r => r.id)
    .attr("transform", r => `translate(${x(xValue(r))},${y(r.throughput_gibs.median)})`)
    .on("click", (event, r) => { svg.node().focus({preventScroll: true}); select(r.id); });
  marks.append("circle").attr("r", 10).attr("fill", "transparent");
  marks.append("path").attr("class", "mark")
    .attr("d", r => d3.symbol().type(shape(r)).size(result.ids.has(r.id) ? 70 : 30)())
    .attr("fill", r => r.control ? "var(--surface-1)" : state.layout !== "overlay" ? color(r)
      : systemFill(svg, `fill-${workload.id}-${r.experiment_id}-${r.codec}`, state.systems.indexOf(r.experiment_id), color(r)))
    .attr("stroke", color)
    .attr("stroke-width", 1.25).attr("opacity", r => result.ids.has(r.id) || r.control ? 1 : .28);
  marks.append("title").text(r => `${experiments.get(r.experiment_id).label} · ${r.codec} / ${r.shuffle} / ${size(r.block_kib)}\n${fmtSignificant(r.throughput_gibs.median)} GiB/s (${fmtSignificant(r.throughput_gibs.min)}–${fmtSignificant(r.throughput_gibs.max)}); ${fmtSignificant(r.compression_fold)}×`);
  const range = g.append("g").attr("class", "range");
  function focused() {
    marks.classed("selected", r => r.id === state.selected);
    range.selectAll("*").remove();
    const r = points.find(r => r.id === state.selected);
    if (!r) return;
    const px = x(xValue(r)), lo = y(r.throughput_gibs.min), hi = y(r.throughput_gibs.max);
    range.append("path").attr("d", `M${px},${lo}V${hi}M${px - 4},${lo}H${px + 4}M${px - 4},${hi}H${px + 4}`);
  }
  chartRedraws.push(focused); focused();
  svg.on("focus", () => { if (!points.some(r => r.id === state.selected)) select(order[0].id, false); });
  svg.on("keydown", event => {
    const i = order.findIndex(r => r.id === state.selected);
    let next;
    if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (i + 1) % order.length;
    else if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (i - 1 + order.length) % order.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = order.length - 1;
    else return;
    event.preventDefault(); select(order[next].id, false);
  });
}

const columns = [
  ["system", "System", r => experiments.get(r.experiment_id).label],
  ["workload", "Input / chunk", r => workloadName(workloads.get(r.workload_id))],
  ["codec", "Codec / setting", r => r.codec], ["shuffle", "Shuffle", r => r.shuffle],
  ["block", "Block", r => r.block_kib, r => size(r.block_kib)],
  ["throughput", "Input GiB/s", r => r.throughput_gibs.median, r => fmt(r.throughput_gibs.median)],
  ["fold", "Fold ×", r => r.compression_fold, r => fmt(r.compression_fold)],
  ["measured", "Measured GiB", r => r.measured_device_gib?.median, r => fmt(r.measured_device_gib?.median)],
  ["estimated", "Estimated GiB", r => memoryValue(r), r => fmt(memoryValue(r))],
  ["repetitions", "Repetitions", r => r.repetitions],
];

function renderTable() {
  const getter = columns.find(c => c[0] === state.sort)[2];
  sortedRows = [...result.candidates].sort((a, b) => {
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
        button.type = "button"; button.setAttribute("aria-label", `Inspect ${experiments.get(row.experiment_id).label}, ${workloadName(workloads.get(row.workload_id))}, ${row.codec}, ${row.shuffle}, ${size(row.block_kib)}`);
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
    tr.append(el("td", result.ids.has(row.id) ? "frontier-label" : null, row.control ? "Raw control" : result.ids.has(row.id) ? "Frontier" : "Candidate"));
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
  chartRedraws.forEach(draw => draw()); highlightTable(); renderDetail();
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
  const panel = $("detail-content"), row = rows.find(r => r.id === state.selected);
  document.body.classList.toggle("has-selection", Boolean(row));
  panel.replaceChildren();
  if (!row) { panel.textContent = "Select a point or table setting to see measurements and provenance."; return; }
  const e = experiments.get(row.experiment_id), w = workloads.get(row.workload_id);
  panel.append(el("h3", null, `${e.label} · ${row.codec}`), el("p", null, `${workloadName(w)} · ${row.shuffle} shuffle · ${size(row.block_kib)} block · level ${row.level}`));
  const visible = result.candidates.some(r => r.id === row.id);
  panel.append(el("p", null, !visible ? "This selection is outside the current filters." : row.control ? "Raw control; excluded from frontier membership." : result.ids.has(row.id) ? "On the current frontier." : "Dominated or missing an objective for the current frontier."));
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
  for (const [key, text] of Object.entries(index.definitions)) {
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
    index = await fetchJson("data/pareto/index.json");
    if (index.version !== 1 || !Array.isArray(index.experiments)) throw new Error("Unsupported Pareto dataset index. Rebuild the report site.");
    const data = await Promise.all(index.experiments.map(e => fetchJson(e.data)));
    for (const [i, d] of data.entries()) {
      if (d.version !== 1 || d.experiment.id !== index.experiments[i].id || !Array.isArray(d.measurements) || !Array.isArray(d.workloads)) throw new Error("Invalid experiment data. Rebuild the report site.");
    }
    rows = data.flatMap(d => d.measurements);
    if (new Set(rows.map(r => r.id)).size !== rows.length) throw new Error("Duplicate measurement identities. Rebuild the report site.");
    experiments = new Map(index.experiments.map(e => [e.id, e]));
    const unique = new Map(data.flatMap(d => d.workloads).map(w => [w.id, w]));
    workloads = new Map([...unique].sort((a,b) => b[1].fill.localeCompare(a[1].fill, "en") || a[1].chunk_kib - b[1].chunk_kib || a[0].localeCompare(b[0], "en")));
    blocks = [...new Set(rows.map(r => r.block_kib))].sort((a,b) => a-b);
    restore(); wireControls(); syncControls(); renderScenarios(); renderMethodology();
    $("workspace").hidden = false; $("load-status").hidden = true; render();
    if (!rows.length) { $("empty").hidden = false; $("empty").textContent = "No retained datasets are listed. Add an experiment to the dataset manifest and rebuild the site."; }
  } catch (error) {
    $("workspace").hidden = true; $("load-status").textContent = `Unable to load Blosc measurements: ${error.message}. Serve the generated site over HTTP and check the data files.`;
    $("load-status").setAttribute("role", "alert"); $("retry").hidden = false;
  }
}

wireThemeToggle(() => { if (state) { renderCharts(); renderLegend(); } });
let lastWidth = 0;
const resize = new ResizeObserver(entries => {
  const width = Math.round(entries[0].contentRect.width);
  if (state && width > 0 && width !== lastWidth) { lastWidth = width; renderCharts(); }
});
resize.observe($("charts"));
$("retry").addEventListener("click", boot);
boot();
