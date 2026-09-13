import {chunkBytes, frontier, isBlosc, measurementsCsv, readState, writeState} from "./microscopy.mjs";

const $ = id => document.getElementById(id);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const format = value => Number.isFinite(value) ? value.toLocaleString("en", {maximumFractionDigits: 3}) : "—";
const size = value => value == null ? "—" : `${format(value / 1024)} KiB`;
const colors = {"blosc-lz4": "var(--series-2)", "blosc-zstd": "var(--series-5)",
  lz4: "var(--series-1)", zstd: "var(--series-3)", none: "var(--series-none)"};
const codecName = codec => ({"blosc-lz4": "Blosc-LZ4 · bitshuffle", "blosc-zstd": "Blosc-Zstd · bitshuffle",
  lz4: "Raw LZ4", zstd: "Raw Zstd", none: "Uncompressed"})[codec] ?? codec;
let rows = [], studies = new Map(), state, result;

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
  highlight();
}

function highlight() {
  for (const tr of $("table-body").children) {
    tr.classList.toggle("selected", tr.dataset.id === state.selected);
    tr.querySelector("button").setAttribute("aria-pressed", String(tr.dataset.id === state.selected));
  }
  d3.selectAll(".point").attr("stroke-width", row => row.id === state.selected ? 4 : result.ids.has(row.id) ? 2 : 0.5);
}

function populateFilters() {
  const unique = values => [...new Set(values)];
  const choices = {
    study: [...studies.values()].map(data => [data.study.id, `${data.study.machine.name} · ${data.phase} · ${data.study.created.slice(0, 10)}`]),
    input: unique(rows.map(row => row.config.input_id)).map(id => [id, rows.find(row => row.config.input_id === id).input_label]),
    backend: unique(rows.map(row => row.config.backend)).map(value => [value, value.toUpperCase()]),
    sink: unique(rows.map(row => row.config.sink)).map(value => [value, value]),
    codec: unique(rows.map(row => row.config.codec.replace(/^blosc-/, ""))).map(value => [value, value === "none" ? "Uncompressed" : value.toUpperCase()]),
    chunk: unique(rows.map(row => row.config.chunk_label)).sort((a, b) => chunkBytes({config: {chunk_label: a}}) - chunkBytes({config: {chunk_label: b}})).map(value => [value, value]),
    block: unique(rows.filter(isBlosc).map(row => row.config.blosc_block_bytes)).sort((a, b) => a - b).map(value => [String(value), size(value)]),
  };
  for (const [key, values] of Object.entries(choices)) {
    $(key).replaceChildren(new Option("All", "all"), ...values.map(([value, label]) => new Option(label, value)));
    $(key).onchange = () => { state[key] = $(key).value; remember(); render(); };
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
    const label = element("span"), mark = element("i");
    mark.style.background = color; label.append(mark, document.createTextNode(codecName(codec))); return label;
  }));
}

function sync() {
  for (const key of ["study", "input", "backend", "sink", "codec", "chunk", "block"]) $(key).value = state[key];
}

function render() {
  result = frontier(rows, state);
  const drifting = result.candidates.filter(row => row.reference.drift).length;
  $("count").textContent = `${result.candidates.length} configurations · ${result.ids.size} observed frontier settings${drifting ? ` · ${drifting} need new reference checks` : ""}`;
  $("empty").hidden = result.candidates.length > 0;
  $("download").disabled = !result.candidates.length;
  renderPlots(); renderTable(); renderDetail(); highlight();
}

function renderPlots() {
  const groups = d3.group(result.candidates, row => row.condition);
  $("plots").replaceChildren();
  const panels = [];
  for (const values of groups.values()) {
    const row = values[0], data = studies.get(row.study_id), panel = element("section", null, "study-plot");
    panel.append(element("h3", `${row.input_label} · ${row.config.backend.toUpperCase()} · ${row.config.sink}`),
      element("p", `${data.study.machine.name} · ${data.phase} · ${data.study.build.revision.slice(0, 7)}`));
    $("plots").append(panel); panels.push([panel, values]);
  }
  const maxRate = d3.max(result.candidates, row => row.throughput.max) || 1;
  const maxFold = d3.max(result.candidates, row => row.compression_fold) || 1;
  for (const [panel, values] of panels) {
    const width = Math.max(240, panel.clientWidth - 28), height = 320;
    const margins = {left: 54, right: 18, top: 15, bottom: 49};
    const x = d3.scaleLinear().domain([0, maxRate * 1.08]).nice().range([margins.left, width - margins.right]);
    const y = d3.scaleLinear().domain([0, maxFold * 1.08]).nice().range([height - margins.bottom, margins.top]);
    const svg = d3.select(panel).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "group")
      .attr("aria-label", `${values[0].input_label}, ${values[0].config.backend}, ${values[0].config.sink}: logical throughput versus compression`);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margins.bottom})`).call(d3.axisBottom(x).ticks(4));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${margins.left},0)`).call(d3.axisLeft(y).ticks(5));
    svg.append("text").attr("class", "axis-title").attr("x", (margins.left + width - margins.right) / 2)
      .attr("y", height - 8).attr("text-anchor", "middle").text("Logical image throughput (GiB/s)");
    svg.append("text").attr("class", "axis-title").attr("transform", `translate(14,${height / 2 - 12}) rotate(-90)`)
      .attr("text-anchor", "middle").text("Logical compression fold (×)");
    const boundary = values.filter(row => result.ids.has(row.id)).sort((a, b) => a.throughput.median - b.throughput.median);
    svg.append("path").datum(boundary).attr("fill", "none").attr("stroke", "var(--text-muted)").attr("stroke-dasharray", "4 4")
      .attr("d", d3.line().x(row => x(row.throughput.median)).y(row => y(row.compression_fold)));
    svg.selectAll(".range").data(values.filter(row => row.count > 1)).join("line").attr("class", "range")
      .attr("x1", row => x(row.throughput.min)).attr("x2", row => x(row.throughput.max))
      .attr("y1", row => y(row.compression_fold)).attr("y2", row => y(row.compression_fold))
      .attr("stroke", row => colors[row.config.codec]).attr("opacity", row => row.reference.drift ? 0.25 : 0.75);
    svg.selectAll(".point").data(values).join("circle").attr("class", "point").attr("tabindex", 0).attr("role", "button")
      .attr("cx", row => x(row.throughput.median)).attr("cy", row => y(row.compression_fold))
      .attr("r", row => result.ids.has(row.id) ? 6 : 4)
      .attr("fill", row => colors[row.config.codec]).attr("stroke", "var(--text-primary)")
      .attr("opacity", row => row.reference.drift ? 0.25 : 0.9)
      .attr("aria-label", row => `${codecName(row.config.codec)}, chunk ${size(chunkBytes(row))}, block ${size(row.config.blosc_block_bytes)}, ${format(row.throughput.median)} logical GiB/s, ${format(row.compression_fold)} fold`)
      .on("click", (_, row) => selected(row.id))
      .on("keydown", (event, row) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selected(row.id); } })
      .append("title").text(row => `${codecName(row.config.codec)}\nChunk ${size(chunkBytes(row))} · block ${size(row.config.blosc_block_bytes)}\n${format(row.throughput.median)} GiB/s · ${format(row.compression_fold)}×\n${row.count} observation(s)${row.needs_confirmation ? " · needs confirmation" : ""}`);
  }
}

function evidence(row) {
  if (row.reference.drift) return "Reference drift";
  if (result.ids.has(row.id)) return row.needs_confirmation ? "Frontier · confirm" : "Observed frontier";
  return row.needs_confirmation ? "Confirm" : "Measured";
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
  target.replaceChildren();
  if (!row) { target.textContent = "Select a point or a table setting to inspect its measurements and pipeline stages."; return; }
  const data = studies.get(row.study_id), raw = row.detail;
  if (!result.candidates.some(item => item.id === row.id)) target.append(element("p", "This configuration is outside the current filters."));
  target.append(element("h3", `${row.input_label} · ${codecName(row.config.codec)}`),
    element("p", `${data.study.machine.name} · ${data.phase} · ${row.config.backend.toUpperCase()} · ${row.config.sink}`),
    pairs([
      ["Chunk", `${size(chunkBytes(row))} · ${raw.image_replay.chunk_shape.join(" × ")}`],
      ["Blosc block request", size(row.config.blosc_block_bytes)], ["Codec level", row.config.level],
      ["Logical throughput", `${format(row.throughput.median)} GiB/s`],
      ["Logical compression", `${format(row.compression_fold)}×`], ["Spatial padding", `${format(row.padding_percent)}%`],
      ["Observations", row.count], ["Observed range", `${format(row.throughput.min)}–${format(row.throughput.max)} GiB/s`],
      ["Nearby reference range", `${format(row.reference.spread_percent)}%`],
      ["All reference range", `${format(row.reference.condition_spread_percent)}%`],
      ["Measured window", `${format(raw.measurement.elapsed_s)} s`],
      ["Final drain", `${format(raw.measurement.drain_s)} s`],
      ["Allowed CPUs / workers", `${data.study.machine.cpu_count} / ${raw.worker_threads}`],
    ]));
  target.append(element("p", row.reference.drift
    ? "Reference measurements drifted within or between groups. Remeasure this input/backend condition before choosing settings."
    : row.needs_confirmation ? "This setting needs more observations before treating its position as stable."
      : "The observed repetition and reference ranges fit the study threshold. This is not a confidence interval."));
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
await load();
