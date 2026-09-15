import {chunkBytes, frontier, isBlosc, measurementsCsv, plottable, plotDomains, readState, resampledFrontier, writeState} from "./microscopy.mjs";
import {plotAxes} from "./charts.js";

const $ = id => document.getElementById(id);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const format = value => Number.isFinite(value) ? value.toLocaleString("en", {maximumFractionDigits: 3}) : "—";
const size = value => value == null ? "—" : `${format(value / 1024)} KiB`;
const colors = {"blosc-lz4": "var(--codec-lz4)", "blosc-zstd": "var(--codec-zstd)",
  lz4: "var(--codec-lz4)", zstd: "var(--codec-zstd)", none: "var(--series-none)"};
const pointShape = codec => codec.startsWith("blosc-") ? d3.symbolTriangle
  : codec === "none" ? d3.symbolSquare : d3.symbolDiamond;
const pointFill = codec => codec === "lz4" || codec === "zstd" ? "var(--surface-1)" : colors[codec];
const codecName = codec => ({"blosc-lz4": "Blosc-LZ4 · bitshuffle", "blosc-zstd": "Blosc-Zstd · bitshuffle",
  lz4: "Raw LZ4", zstd: "Raw Zstd", none: "Uncompressed"})[codec] ?? codec;
let rows = [], studies = new Map(), state, result, outsideView = new Set();

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function remember(replace = false) {
  const url = `${location.pathname}?${writeState(state)}${location.hash}`;
  if (url !== location.pathname + location.search + location.hash) history[replace ? "replaceState" : "pushState"](null, "", url);
}

function selected(id) {
  state.selected = id;
  remember();
  renderDetail();
  $("detail").scrollTop = 0;
  if (matchMedia("(max-width: 1100px)").matches) $("detail").scrollIntoView({block: "start"});
  highlight();
}

function highlight() {
  for (const tr of $("table-body").children) {
    tr.classList.toggle("selected", tr.dataset.id === state.selected);
    tr.querySelector("button").setAttribute("aria-pressed", String(tr.dataset.id === state.selected));
  }
  d3.selectAll(".point").classed("selected", row => row.id === state.selected)
    .attr("aria-pressed", row => String(row.id === state.selected));
}

function populateFilters() {
  const unique = values => [...new Set(values)];
  const choices = {
    study: [...studies.values()].map(data => [data.study.id, `${data.study.machine.name} · ${data.phase} · ${data.study.created.slice(0, 10)}`]),
    input: unique(rows.map(row => row.config.input_id)).map(id => [id, rows.find(row => row.config.input_id === id).input_label]),
    backend: unique(rows.map(row => row.config.backend)).map(value => [value, value.toUpperCase()]),
    sink: unique(rows.map(row => row.config.sink)).map(value => [value, value]),
    codec: unique(rows.map(row => row.config.codec)).map(value => [value,
      value === "none" ? "Uncompressed" : value.startsWith("blosc-") ? value : `${value} (raw)`]),
    chunk: unique(rows.map(row => row.config.chunk_label)).sort((a, b) => chunkBytes({config: {chunk_label: a}}) - chunkBytes({config: {chunk_label: b}})).map(value => [value, value]),
    block: unique(rows.filter(isBlosc).map(row => row.config.blosc_block_bytes)).sort((a, b) => a - b).map(value => [String(value), size(value)]),
  };
  for (const [key, values] of Object.entries(choices)) {
    $(key).replaceChildren(new Option("All", "all"), ...values.map(([value, label]) => new Option(label, value)));
    $(key).onchange = () => { state[key] = $(key).value; remember(); render(); };
  }
  $("axes").onchange = () => { state.axes = $("axes").value; remember(); render(); };
  for (const [id, extent] of [["fit-frontier", "frontier"], ["show-all", "all"]]) {
    $(id).onclick = () => { state.extent = extent; remember(); render(); };
  }
  $("filters").onsubmit = event => event.preventDefault();
  $("reset").onclick = () => { state = readState("", rows); remember(); sync(); render(); };
  $("download").onclick = () => {
    const blob = new Blob([measurementsCsv(result.candidates, result.ids)], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob), link = element("a");
    link.href = url; link.download = "microscopy-filtered-measurements.csv"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $("legend").replaceChildren(...Object.entries(colors).map(([codec, color]) => {
    const label = element("span");
    const mark = d3.create("svg").attr("viewBox", "-11 -11 22 22").attr("aria-hidden", "true");
    mark.append("path").attr("d", d3.symbol().type(pointShape(codec)).size(65)())
      .attr("fill", pointFill(codec)).attr("stroke", color).attr("stroke-width", 1.5);
    label.append(mark.node(), document.createTextNode(codecName(codec))); return label;
  }));
}

function sync() {
  for (const key of ["study", "input", "backend", "sink", "codec", "chunk", "block", "axes"]) $(key).value = state[key];
}

function render() {
  result = frontier(rows, state);
  const drifting = result.candidates.filter(row => row.reference.drift).length;
  $("count").textContent = `${result.candidates.length} configurations · ${result.ids.size} observed frontier settings${drifting ? ` · ${drifting} have reference variation` : ""}`;
  $("empty").hidden = result.candidates.length > 0;
  $("download").disabled = !result.candidates.length;
  $("fit-frontier").setAttribute("aria-pressed", String(state.extent === "frontier"));
  $("fit-frontier").disabled = !result.ids.size;
  $("show-all").setAttribute("aria-pressed", String(state.extent === "all"));
  renderPlots(); renderTable(); renderDetail(); highlight();
  $("axis-note").textContent = ({panel: "Axes fit each panel separately; their limits differ.",
    input: "Panels of the same image input share axis limits.",
    all: "All panels share axis limits."})[state.axes] +
    (state.extent === "frontier" && result.ids.size
      ? ` Frontier view: ${outsideView.size} other points lie outside the plot limits; the table and CSV retain all configurations.` : "") +
    " Compression fold uses a log scale. Padding can make uncompressed output larger than the logical image, giving a fold below 1." +
    (result.candidates.some(row => row.uncertainty)
      ? " Rings mark additional settings on the frontier in at least 5% of whole-round resamples. Frequencies use all selected settings in each study condition, including settings hidden by filters. Bars show observed min–max; approximate bootstrap intervals are in details."
      : "");
}

function pointDescription(row) {
  const data = studies.get(row.study_id), raw = row.detail;
  return [
    `${row.input_label} · ${row.config.dtype}`,
    `${data.study.machine.name} · ${row.config.backend.toUpperCase()} · ${row.config.sink}`,
    `${codecName(row.config.codec)}${row.config.codec === "none" ? "" : ` · level ${row.config.level}`}`,
    `Chunk ${size(chunkBytes(row))} · shape ${raw.image_replay.chunk_shape.join(" × ")}`,
    isBlosc(row) ? `Blosc block request ${size(row.config.blosc_block_bytes)}` : "No Blosc blocks",
    `Logical throughput ${format(row.throughput.median)} GiB/s`,
    `${row.count} observation(s) · min–max ${format(row.throughput.min)}–${format(row.throughput.max)} GiB/s`,
    `Logical compression fold ${format(row.compression_fold)}× · padding ${format(row.padding_percent)}%`,
    `${raw.worker_threads} workers · ${data.study.machine.cpu_count} allowed CPUs`,
    row.compression_range ? `Observed fold ${format(row.compression_range.min)}–${format(row.compression_range.max)}×` : null,
    row.uncertainty ? `Frontier in ${format(100 * row.uncertainty.frontier_frequency)}% of ${row.uncertainty.draws} round resamples · ${row.uncertainty.configurations} settings` : null,
    row.count === 1 ? "Run variation unknown: one observation" : evidence(row),
    `${data.phase} · source ${data.study.build.revision.slice(0, 7)}`,
  ].filter(Boolean).join("\n");
}

function renderPlots() {
  const groups = d3.group(result.candidates, row => row.condition);
  $("plots").replaceChildren();
  const panels = [];
  outsideView = new Set();
  const scaleNote = {panel: "Fitted axes", input: "Axes match this input", all: "Axes match all panels"}[state.axes];
  for (const candidates of groups.values()) {
    const row = candidates[0], data = studies.get(row.study_id), panel = element("section", null, "study-plot");
    const values = candidates.filter(plottable), unavailable = candidates.length - values.length;
    panel.append(element("h3", `${row.input_label} · ${row.config.dtype} · ${row.config.backend.toUpperCase()} · ${row.config.sink}`),
      element("p", `${data.study.machine.name} · ${data.phase} · ${data.study.build.revision.slice(0, 7)}`),
      element("p", `${scaleNote}${unavailable ? ` · ${unavailable} unavailable on these axes` : ""}`, "plot-scale"));
    $("plots").append(panel); panels.push([panel, values, row]);
  }
  for (const [index, [panel, values, first]] of panels.entries()) {
    const scope = state.axes === "panel" ? values : state.axes === "input"
      ? result.candidates.filter(row => row.config.input_id === first.config.input_id) : result.candidates;
    const boundaryScope = scope.filter(row => result.ids.has(row.id) || resampledFrontier(row));
    const domains = plotDomains(state.extent === "frontier" && boundaryScope.length ? boundaryScope : scope);
    const inView = row => row.compression_fold >= domains.fold[0] && row.compression_fold <= domains.fold[1]
      && row.throughput.median >= domains.throughput[0] && row.throughput.median <= domains.throughput[1];
    for (const row of values) if (!inView(row)) outsideView.add(row.id);
    const width = Math.max(240, panel.clientWidth - 28), height = 300;
    const margin = {left: 56, top: 18, width: width - 76, height: height - 70};
    const x = d3.scaleLog().domain(domains.fold).range([0, margin.width]);
    const y = d3.scaleLinear().domain(domains.throughput).range([margin.height, 0]);
    const svg = d3.select(panel).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "group")
      .attr("aria-label", `${first.input_label}, ${first.config.dtype}, ${first.config.backend}, ${first.config.sink}: logical compression fold on the logarithmic horizontal axis, logical throughput on the vertical axis. ${scaleNote}.`);
    const plot = plotAxes(svg, x, y, {...margin,
      xLabel: "Logical compression fold (×)", yLabel: "Logical throughput (GiB/s)",
      xTicks: domains.fold[1] / domains.fold[0] < 10 ? d3.ticks(...domains.fold, 4) : null});
    const clipId = `microscopy-plot-${index}`;
    svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
      .attr("width", margin.width).attr("height", margin.height);
    const points = plot.append("g").attr("clip-path", `url(#${clipId})`);
    if (!values.length) {
      plot.append("text").attr("class", "plot-empty").attr("x", margin.width / 2).attr("y", margin.height / 2)
        .attr("text-anchor", "middle").text("No values available on these axes");
      continue;
    }
    if (domains.fold[0] <= 1 && domains.fold[1] >= 1) {
      plot.append("line").attr("class", "fold-baseline").attr("x1", x(1)).attr("x2", x(1))
        .attr("y1", 0).attr("y2", margin.height).append("title").text("1×: output equals logical input size");
    }
    const boundary = values.filter(row => result.ids.has(row.id)).sort((a, b) => a.compression_fold - b.compression_fold);
    points.append("path").datum(boundary).attr("class", "frontier-line")
      .attr("d", d3.line().x(row => x(row.compression_fold)).y(row => y(row.throughput.median)));
    points.selectAll(".range").data(values.filter(row => row.count > 1
      && Number.isFinite(row.throughput.min) && Number.isFinite(row.throughput.max)))
      .join("line").attr("class", "range")
      .attr("x1", row => x(row.compression_fold)).attr("x2", row => x(row.compression_fold))
      .attr("y1", row => y(row.throughput.min)).attr("y2", row => y(row.throughput.max))
      .attr("stroke", row => colors[row.config.codec]).attr("opacity", row => result.ids.has(row.id) || resampledFrontier(row) ? 0.75 : 0.25);
    points.selectAll(".fold-range").data(values.filter(row => row.count > 1 && row.compression_range))
      .join("line").attr("class", "fold-range")
      .attr("x1", row => x(row.compression_range.min)).attr("x2", row => x(row.compression_range.max))
      .attr("y1", row => y(row.throughput.median)).attr("y2", row => y(row.throughput.median))
      .attr("stroke", row => colors[row.config.codec]).attr("opacity", 0.6);
    const marks = points.selectAll(".point")
      .data([...values].sort((a, b) => Number(result.ids.has(a.id)) - Number(result.ids.has(b.id))))
      .join("g").attr("class", row => result.ids.has(row.id) ? "point frontier" : resampledFrontier(row) ? "point resampled-frontier" : "point")
      .attr("data-id", row => row.id).attr("tabindex", row => inView(row) ? 0 : -1).attr("role", "button")
      .attr("transform", row => `translate(${x(row.compression_fold)},${y(row.throughput.median)})`)
      .attr("aria-label", pointDescription)
      .on("click", (_, row) => selected(row.id))
      .on("keydown", (event, row) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selected(row.id); } });
    marks.append("circle").attr("r", 10).attr("fill", "transparent");
    marks.filter(row => !result.ids.has(row.id) && resampledFrontier(row)).append("circle")
      .attr("class", "frontier-ring").attr("r", 7).attr("fill", "none")
      .attr("stroke", row => colors[row.config.codec]).attr("stroke-width", 1.5);
    marks.append("path").attr("class", "mark")
      .attr("d", row => d3.symbol().type(pointShape(row.config.codec)).size(result.ids.has(row.id) ? 80 : 34)())
      .attr("fill", row => pointFill(row.config.codec)).attr("stroke", row => colors[row.config.codec])
      .attr("stroke-width", 1.4).attr("opacity", row => result.ids.has(row.id) ? 1 : resampledFrontier(row) ? 0.75 : 0.35);
    marks.append("title").text(pointDescription);
  }
}

function evidence(row) {
  if (row.reference.drift) return "Reference variation";
  if (result.ids.has(row.id)) return row.count === 1 ? "Observed frontier · 1 observation" : "Observed frontier";
  if (resampledFrontier(row)) return "Frontier in some resamples";
  return row.count === 1 ? "1 observation · variation unknown" : `${row.count} observations`;
}

function renderTable() {
  const ordered = [...result.candidates].sort((a, b) => Number(result.ids.has(b.id)) - Number(result.ids.has(a.id)) || b.throughput.median - a.throughput.median);
  $("table-body").replaceChildren(...ordered.map(row => {
    const tr = element("tr"), setting = element("td"), button = element("button", codecName(row.config.codec), "setting-button");
    button.type = "button"; button.onclick = () => selected(row.id);
    setting.append(button, element("small", row.input_label)); tr.dataset.id = row.id;
    tr.append(setting, element("td", `${row.config.backend.toUpperCase()} / ${row.config.sink}`),
      element("td", size(chunkBytes(row))), element("td", size(row.config.blosc_block_bytes)),
      element("td", format(row.throughput.median)), element("td", format(row.compression_fold)),
      element("td", evidence(row), row.reference.drift ? "drift" : result.ids.has(row.id) ? "frontier-label" : null));
    return tr;
  }));
}

function pairs(values) {
  const list = element("dl");
  for (const [label, value] of values) list.append(element("dt", label), element("dd", value));
  return list;
}

function renderDetail() {
  const target = $("detail-content"), row = rows.find(item => item.id === state.selected);
  document.body.classList.toggle("has-selection", Boolean(row));
  target.replaceChildren();
  if (!row) { target.textContent = "Select a point or a table setting to inspect its measurements and pipeline stages."; return; }
  const data = studies.get(row.study_id), raw = row.detail;
  if (!result.candidates.some(item => item.id === row.id)) target.append(element("p", "This configuration is outside the current filters."));
  else if (outsideView.has(row.id)) target.append(element("p", "This configuration is outside the frontier view. Choose Show all points to see its marker."));
  target.append(element("h3", `${row.input_label} · ${codecName(row.config.codec)}`),
    element("p", `${data.study.machine.name} · ${data.phase} · ${row.config.backend.toUpperCase()} · ${row.config.sink}`),
    pairs([
      ["Chunk", `${size(chunkBytes(row))} · ${raw.image_replay.chunk_shape.join(" × ")}`],
      ["Blosc block request", size(row.config.blosc_block_bytes)], ["Codec level", row.config.level],
      ["Data type", row.config.dtype],
      ["Logical throughput", `${format(row.throughput.median)} GiB/s`],
      ["Logical compression fold", `${format(row.compression_fold)}×`], ["Spatial padding", `${format(row.padding_percent)}%`],
      ["Observations", row.count], ["Observed range", `${format(row.throughput.min)}–${format(row.throughput.max)} GiB/s`],
      ["Nearby reference range", `${format(row.reference.spread_percent)}%`],
      ["All reference range", `${format(row.reference.condition_spread_percent)}%`],
      ["Measured window", `${format(raw.measurement.elapsed_s)} s`],
      ["Final drain", `${format(raw.measurement.drain_s)} s`],
      ["Allowed CPUs / workers", `${data.study.machine.cpu_count} / ${raw.worker_threads}`],
    ]));
  if (data.study.sink_options?.tmpdir && row.config.sink === "fs") {
    target.append(pairs([["Filesystem root", data.study.sink_options.tmpdir]]));
    target.append(element("p", "Throughput includes the final pipeline drain and close through this filesystem. The result depends on platform buffering and does not establish crash-durable disk bandwidth."));
  }
  target.append(element("p", row.count === 1
    ? "One observation cannot characterize run variation."
    : "Bars show the observed min–max range; repetition count and reference variation remain visible."));
  if (row.reference.drift) target.append(element("p", "Reference throughput varied across this session. The observed ranges may not describe variation between sessions."));
  if (row.uncertainty) {
    const uncertainty = row.uncertainty;
    target.append(element("h3", "Variation across rounds"), pairs([
      ["Rounds", uncertainty.rounds],
      ["Throughput interval", `${format(uncertainty.throughput.lower)}–${format(uncertainty.throughput.upper)} GiB/s`],
      ["Compression fold interval", `${format(uncertainty.compression_fold.lower)}–${format(uncertainty.compression_fold.upper)}×`],
      ["Frontier in round resamples", `${format(100 * uncertainty.frontier_frequency)}%`],
      ["Compared settings", uncertainty.configurations],
    ]), element("p", "Approximate 95% bootstrap intervals use paired whole rounds. Frontier frequency is resampling support across all selected settings in this condition, including hidden settings; it is not a posterior probability or a simultaneous confidence bound."));
  }
  target.append(element("h3", `Pipeline stages · ${row.detail_execution}`), element("p", "Stage rates use each stage’s own input/output bytes. Intervals overlap and do not sum to elapsed time."));
  const scroll = element("div", null, "table-scroll"), table = element("table"), head = element("thead"), header = element("tr"), body = element("tbody");
  for (const title of ["Stage", "Avg ms", "In GiB/s", "Out GiB/s"]) { const th = element("th", title); th.scope = "col"; header.append(th); }
  head.append(header);
  for (const [name, stage] of Object.entries(raw.stages ?? {})) {
    const tr = element("tr"); tr.append(element("td", name), element("td", format(stage.avg_ms)),
      element("td", format(stage.in_gibs)), element("td", format(stage.out_gibs))); body.append(tr);
  }
  table.append(head, body); scroll.append(table); target.append(scroll);
  const source = element("a", `Source ${data.study.build.revision.slice(0, 7)}`);
  source.href = `https://github.com/acquire-project/chucky/tree/${encodeURIComponent(data.study.build.revision)}`;
  const archive = element("a", "Retained raw observations"); archive.href = data.study.archive;
  const links = element("p"); links.append(source, document.createTextNode(" · "), archive); target.append(links);
  const details = element("details"), summary = element("summary", "Commands, input hashes, and replay details");
  details.append(summary, element("pre", JSON.stringify({id: row.id, executions: row.samples,
    references: row.reference, binary_sha256: data.study.build.executable_sha256,
    image_input: raw.image_input, replay: raw.image_replay, command: raw.command,
    measurement: raw.measurement}, null, 2))); target.append(details);
}

function renderArchives() {
  $("archives").replaceChildren(element("h3", "Retained sources"));
  for (const data of studies.values()) {
    const paragraph = element("p"), link = element("a", `${data.study.machine.name} · ${data.phase} · ${data.study.created.slice(0, 10)}`);
    link.href = data.study.archive;
    paragraph.append(link, document.createTextNode(` · ${data.counts.configurations} configurations, ${data.counts.executions} executions · source ${data.study.build.revision.slice(0, 7)}`));
    $("archives").append(paragraph);
  }
}

async function load() {
  $("retry").hidden = true;
  try {
    const index = await getJson("data/microscopy/index.json");
    if (index.version !== 1 || !Array.isArray(index.studies)) throw new Error("Unsupported study index");
    if (!index.studies.length) {
      $("load-status").textContent = "No retained microscopy study has been published yet. The discovery definition is available below.";
      return;
    }
    const datasets = await Promise.all(index.studies.map(item => getJson(item.file)));
    studies = new Map(datasets.map(data => [data.study.id, data]));
    rows = datasets.flatMap(data => data.measurements);
    state = readState(location.search, rows);
    populateFilters(); sync(); renderArchives();
    $("load-status").hidden = true; $("workspace").hidden = false;
    render(); remember(true);
  } catch (error) {
    $("load-status").hidden = false;
    $("load-status").textContent = `Could not load microscopy measurements: ${error.message}`;
    $("retry").hidden = false;
  }
}

$("retry").onclick = load;
window.addEventListener("popstate", () => { if (rows.length) { state = readState(location.search, rows); sync(); render(); } });
let resize;
window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => { if (result) { renderPlots(); highlight(); } }, 120); });
wireThemeToggle(() => { if (result) { renderPlots(); highlight(); } });
await load();
