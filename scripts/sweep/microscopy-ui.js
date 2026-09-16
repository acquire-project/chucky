import {chunkBytes, frontier, isBlosc, machinePanels, measurementsCsv, plottable, plotDomains, readState, reportRows, resampledFrontier, writeState} from "./microscopy.mjs";
import {plotAxes} from "./charts.js";
import {machineChoices, syncMachines} from "./pareto-controls.js";
import {hardwareDetails, runDates} from "./pareto-metadata.mjs";

const $ = id => document.getElementById(id);
const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const format = value => Number.isFinite(value) ? value.toLocaleString("en", {maximumFractionDigits: 3}) : "—";
const unique = values => [...new Set(values)];
const dtypeName = value => ({u8: "uint8", u16: "uint16", f32: "float32"})[value] ?? value;
const sinkName = value => value === "fs" ? "Filesystem" : value === "discard" ? "Discard" : value;
const spatialShards = replay => replay.shape.slice(1).reduce((count, length, index) =>
  count * Math.ceil(length / (replay.chunk_shape[index + 1] * replay.chunks_per_shard[index + 1])), 1);
const size = value => value == null ? "—" : `${format(value / 1024)} KiB`;
const colors = {"blosc-lz4": "var(--codec-lz4)", "blosc-zstd": "var(--codec-zstd)",
  lz4: "var(--codec-lz4)", zstd: "var(--codec-zstd)", none: "var(--series-none)"};
const pointShape = codec => ({"blosc-lz4": d3.symbolTriangle, "blosc-zstd": d3.symbolSquare,
  lz4: d3.symbolDiamond, zstd: d3.symbolCircle, none: d3.symbolCross})[codec];
const pointFill = (codec, color = colors[codec]) => codec === "lz4" || codec === "zstd" ? "var(--surface-1)" : color;
const codecName = codec => ({"blosc-lz4": "Blosc-LZ4 · bitshuffle", "blosc-zstd": "Blosc-Zstd · bitshuffle",
  lz4: "Raw LZ4", zstd: "Raw Zstd", none: "Uncompressed"})[codec] ?? codec;
let rows = [], studies = new Map(), previews = new Map(), entropies = new Map(), inputColors = new Map(), state, result, outsideView = new Set();
const inputKey = row => `${row.config.image_asset_id}:${row.detail.image_input.pack_sha256}`;
const previewFor = row => previews.get(inputKey(row));
const entropyValue = sample => sample.pixel_entropy_bits.toFixed(2);
const entropySampleLimited = sample => sample.unique_values === sample.sample_pixels
  && sample.sample_pixels < sample.shape.reduce((count, length) => count * length, 1);
const pointColor = row => state.input === "all" ? inputColors.get(row.config.input_id) : colors[row.config.codec];

function thumbnail(row) {
  const preview = previewFor(row), frame = element("span", null, "dataset-thumbnail");
  if (preview) {
    const image = element("img");
    image.src = preview.file; image.alt = preview.name; image.width = 128; image.height = 128;
    frame.append(image);
  } else frame.textContent = dtypeName(row.config.dtype);
  return frame;
}

function inputPreview(input) {
  if (input !== "all") return thumbnail(rows.find(row => row.config.input_id === input));
  const mosaic = element("span", null, "dataset-thumbnail dataset-mosaic");
  for (const id of unique(rows.map(row => row.config.input_id))) {
    const preview = previewFor(rows.find(row => row.config.input_id === id));
    if (!preview) continue;
    const image = element("img"); image.src = preview.file; image.alt = ""; image.width = 128; image.height = 128;
    mosaic.append(image);
  }
  return mosaic;
}

function changeInput(input) {
  state.input = input; state.selected = null;
  remember(); render();
}

function emphasizeInput(input) {
  d3.selectAll(".study-plot [data-input]").classed("dataset-dimmed", function() {
    return state.input === "all" && input != null && this.dataset.input !== input;
  });
}

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
  if (matchMedia("(max-width: 1100px)").matches) $("close-detail").focus({preventScroll: true});
  highlight();
}

function closeDetail() {
  const id = state.selected;
  state.selected = null;
  remember(); renderDetail(); highlight();
  const point = document.querySelector(`.point[data-id="${CSS.escape(id ?? "")}"][tabindex="0"]`);
  (point ?? $("plots")).focus({preventScroll: true});
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
  const inputs = unique(rows.map(row => row.config.input_id));
  const machines = unique(rows.map(row => row.machine));
  machineChoices($("systems"), machines.map(machine => {
    const available = rows.filter(row => row.machine === machine);
    const sources = unique(available.map(row => row.study_id)).map(id => studies.get(id).study);
    return {id: machine, label: machine,
      details: hardwareDetails(sources.map(study => ({hardware: study.machine, specs: study.machine_specs})))};
  }), selected => {
    state.machines = selected; state.selected = null;
    remember(); render();
  });
  $("settings").open = ["backend", "sink", "codec", "chunk", "block"].some(key => state[key] !== "all") || state.axes !== "input";
  inputColors = new Map(inputs.map((id, index) => [id, `var(--dataset-${index % 7})`]));
  const choices = {
    input: inputs.map(id => [id, rows.find(row => row.config.input_id === id).input_label]),
    backend: unique(rows.map(row => row.config.backend)).map(value => [value, value.toUpperCase()]),
    sink: unique(rows.map(row => row.config.sink)).map(value => [value, sinkName(value)]),
    codec: unique(rows.map(row => row.config.codec)).map(value => [value,
      value === "none" ? "Uncompressed" : value.startsWith("blosc-") ? value : `${value} (raw)`]),
    chunk: unique(rows.map(row => row.config.chunk_label)).sort((a, b) => chunkBytes({config: {chunk_label: a}}) - chunkBytes({config: {chunk_label: b}})).map(value => [value, size(chunkBytes({config: {chunk_label: value}}))]),
    block: unique(rows.filter(isBlosc).map(row => row.config.blosc_block_bytes)).sort((a, b) => a - b).map(value => [String(value), size(value)]),
  };
  for (const [key, values] of Object.entries(choices)) {
    $(key).replaceChildren(new Option(key === "input" ? "All datasets" : "All", "all"), ...values.map(([value, label]) => new Option(label, value)));
    $(key).onchange = () => {
      state[key] = $(key).value;
      if (key === "input") state.selected = null;
      remember(); render();
    };
  }
  $("dataset-tabs").replaceChildren(...[...inputs, "all"].map(id => {
    const inputRows = id === "all" ? rows : rows.filter(row => row.config.input_id === id), row = inputRows[0];
    const button = element("button"), copy = element("span", null, "dataset-tab-copy");
    button.type = "button"; button.dataset.input = id;
    copy.append(element("span", id === "all" ? "All datasets" : row.input_label),
      element("small", id === "all" ? `${inputs.length} inputs together` : `${dtypeName(row.config.dtype)} · ${inputRows.length} settings`));
    button.append(inputPreview(id), copy);
    button.onclick = () => changeInput(id);
    button.onmouseenter = button.onfocus = () => emphasizeInput(id === "all" ? null : id);
    button.onmouseleave = button.onblur = () => emphasizeInput(null);
    return button;
  }));
  $("dataset-legend").replaceChildren(...inputs.map(id => {
    const row = rows.find(row => row.config.input_id === id), button = element("button");
    button.type = "button"; button.dataset.input = id;
    button.style.setProperty("--dataset-color", inputColors.get(id));
    const swatch = element("span", null, "dataset-swatch"); swatch.setAttribute("aria-hidden", "true");
    button.append(thumbnail(row), swatch, element("span", row.input_label));
    button.setAttribute("aria-label", `Show ${row.input_label}`);
    button.onclick = () => changeInput(id);
    button.onmouseenter = button.onfocus = () => emphasizeInput(id);
    button.onmouseleave = button.onblur = () => emphasizeInput(null);
    return button;
  }));
  $("axes").onchange = () => { state.axes = $("axes").value; remember(); render(); };
  for (const [id, extent] of [["fit-frontier", "frontier"], ["show-all", "all"]]) {
    $(id).onclick = () => { state.extent = extent; remember(); render(); };
  }
  $("filters").onsubmit = event => event.preventDefault();
  $("reset-filters").onclick = () => {
    state = readState(writeState({input: state.input}), rows);
    remember(); render();
  };
  $("close-detail").onclick = closeDetail;
  $("download").onclick = () => {
    const blob = new Blob([measurementsCsv(result.candidates, result.ids)], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob), link = element("a");
    link.href = url; link.download = "microscopy-filtered-measurements.csv"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $("report-summary").textContent = `${machines.length} machines · ${inputs.length} image inputs · ${rows.length} configurations`;
  $("resampling-note").hidden = !rows.some(row => row.uncertainty);
}

function renderLegend() {
  $("legend").replaceChildren(...Object.entries(colors).map(([codec, codecColor]) => {
    const color = state.input === "all" ? "var(--text-secondary)" : codecColor, label = element("span");
    const mark = d3.create("svg").attr("viewBox", "-11 -11 22 22").attr("aria-hidden", "true");
    mark.append("path").attr("d", d3.symbol().type(pointShape(codec)).size(65)())
      .attr("fill", pointFill(codec, color)).attr("stroke", color).attr("stroke-width", 1.5);
    label.append(mark.node(), document.createTextNode(codec.startsWith("blosc-") ? codec : codecName(codec)));
    return label;
  }));
  $("dataset-legend").hidden = state.input !== "all";
  $("reading-note").textContent = (state.input === "all"
    ? "Each dataset has its own frontier. "
    : "Connected points form the observed frontier. ") + "Bars show observed min–max, not confidence intervals.";
}

function sync() {
  renderEntropy();
  syncMachines($("systems"), state.machines);
  for (const key of ["input", "backend", "sink", "codec", "chunk", "block", "axes"]) $(key).value = state[key];
  const machineRows = rows.filter(row => state.machines.includes(row.machine));
  for (const button of $("dataset-tabs").children) {
    const input = button.dataset.input;
    button.setAttribute("aria-pressed", String(input === state.input));
    const available = machineRows.filter(row => input === "all" || row.config.input_id === input);
    button.querySelector("small").textContent = input === "all"
      ? `${unique(available.map(row => row.config.input_id)).length} inputs together`
      : available.length ? `${dtypeName(available[0].config.dtype)} · ${available.length} settings` : "Not measured";
  }
  $("input-thumbnail").replaceChildren(inputPreview(state.input));
}

function render() {
  const position = {left: scrollX, top: scrollY, behavior: "instant"};
  sync();
  result = frontier(rows, state);
  const observations = unique(result.candidates.map(row => row.count)).sort((a, b) => a - b);
  const repeats = observations.length === 1 ? observations[0] : `${observations[0]}–${observations.at(-1)}`;
  const machineRows = rows.filter(row => state.machines.includes(row.machine));
  const inputRows = machineRows.filter(row => state.input === "all" || row.config.input_id === state.input);
  const inputs = unique(inputRows.map(row => row.config.input_id));
  const cpuInputs = unique(inputRows.filter(row => row.config.backend === "cpu").map(row => row.config.input_id));
  const coverage = cpuInputs.length === inputs.length ? "median throughput" : cpuInputs.length
    ? `CPU data for ${cpuInputs.length}/${inputs.length} inputs` : "CPU measurements unavailable";
  $("coverage").textContent = result.candidates.length ? `${repeats} observations · ${coverage}` : "No matching measurements";
  $("count").textContent = `${result.ids.size} observed frontier settings · ${result.candidates.length} measured`;
  $("table-count").textContent = `(${result.candidates.length})`;
  const filters = ["backend", "sink", "codec", "chunk", "block"].filter(key => state[key] !== "all");
  $("active-filters").hidden = !filters.length;
  $("active-filters").textContent = filters.map(key => `${({chunk: "Chunk", block: "Block"})[key] ?? ""} ${$(key).selectedOptions[0]?.textContent ?? state[key]}`.trim()).join(" · ");
  $("empty").hidden = result.candidates.length > 0;
  $("empty").textContent = state.machines.length ? "No configurations match these filters." : "Select a machine to display its measurements.";
  $("download").disabled = !result.candidates.length;
  $("fit-frontier").setAttribute("aria-pressed", String(state.extent === "frontier"));
  $("fit-frontier").disabled = !result.ids.size;
  $("show-all").setAttribute("aria-pressed", String(state.extent === "all"));
  renderLegend(); renderPlots(); renderTable(); renderDetail(); highlight();
  $("axis-note").textContent = ({panel: "Axes fit each panel; limits differ.",
    input: "Visible panels share axis limits.",
    all: "Axes stay the same across datasets with these filters."})[state.axes] +
    " Compression fold uses a log scale. Padding can make uncompressed output larger, giving a fold below 1." +
    (state.extent === "frontier" && outsideView.size
      ? ` ${outsideView.size} other ${outsideView.size === 1 ? "setting lies" : "settings lie"} outside the plot limits; the table and CSV include them.` : "") +
    (result.candidates.some(row => row.uncertainty)
      ? " Rings indicate frontier support in round resamples across all selected settings. See details for its scope." : "");
  scrollTo(position);
}

function pointDescription(row) {
  const data = studies.get(row.study_id), raw = row.detail;
  return [
    `${row.input_label} · ${dtypeName(row.config.dtype)}`,
    `${data.study.machine.name} · ${row.config.backend.toUpperCase()} · ${sinkName(row.config.sink)} (${row.config.sink})`,
    `${codecName(row.config.codec)}${row.config.codec === "none" ? "" : ` · level ${row.config.level}`}`,
    `Chunk ${size(chunkBytes(row))} · shape ${raw.image_replay.chunk_shape.join(" × ")}`,
    isBlosc(row) ? `Blosc block request ${size(row.config.blosc_block_bytes)}` : "No Blosc blocks",
    `Spatial shard files ${spatialShards(raw.image_replay)} · chunks per shard ${raw.image_replay.chunks_per_shard.join(" × ")}`,
    `Logical throughput ${format(row.throughput.median)} GiB/s`,
    `${row.count} observation(s) · min–max ${format(row.throughput.min)}–${format(row.throughput.max)} GiB/s`,
    `Logical compression fold ${format(row.compression_fold)}× · padding ${format(row.padding_percent)}%`,
    `${raw.worker_threads} workers · ${raw.execution_resources?.cpu_affinity?.length ?? data.study.machine.cpu_count} allowed CPU threads`,
    data.study.machine.cpu_topology?.allowed_physical_cores != null
      ? `${data.study.machine.cpu_topology.allowed_physical_cores} allocated physical CPU cores` : null,
    data.study.build.build_settings?.CHUCKY_IO_WORKERS
      ? `${data.study.build.build_settings.CHUCKY_OUTPUT_BUFFERS} output buffers · ${data.study.build.build_settings.CHUCKY_IO_WORKERS} I/O workers` : null,
    row.compression_range ? `Observed fold ${format(row.compression_range.min)}–${format(row.compression_range.max)}×` : null,
    row.uncertainty ? `Frontier in ${format(100 * row.uncertainty.frontier_frequency)}% of ${row.uncertainty.draws} round resamples · ${row.uncertainty.configurations} settings` : null,
    row.count === 1 ? "Run variation unknown: one observation" : evidence(row),
    `Source ${data.study.build.revision.slice(0, 7)}`,
  ].filter(Boolean).join("\n");
}

function renderPlots() {
  outsideView = new Set();
  const scaleNote = {panel: "Independent axes", input: "Matched panels", all: "Fixed across datasets"}[state.axes];
  const matched = state.axes === "all" ? frontier(rows, {...state, input: "all"}) : result;
  const hosts = d3.select("#plots").selectAll(".machine-section")
    .data(machinePanels(result.candidates, state.backend), host => host.machine)
    .join(enter => {
      const host = enter.append("section").attr("class", "machine-section");
      host.append("h2").attr("class", "machine-heading");
      host.append("div").attr("class", "machine-sinks");
      return host;
    }).attr("data-machine", host => host.machine);
  hosts.select(".machine-heading").text(host => host.machine);
  hosts.each(function(host) {
    const sinks = d3.select(this).select(".machine-sinks").selectAll(".sink-pair")
      .data(host.sinks, sink => sink.sink).join(enter => {
        const sink = enter.append("section").attr("class", "sink-pair");
        sink.append("h3").attr("class", "sink-heading");
        sink.append("div").attr("class", "study-plots");
        return sink;
      }).attr("data-sink", sink => sink.sink);
    sinks.select(".sink-heading").text(sink => `${sinkName(sink.sink)} output`);
    sinks.select(".study-plots").classed("single-backend", state.backend !== "all")
      .selectAll(".study-plot").data(sink => sink.panels, panel => panel.backend)
      .join(enter => {
        const panel = enter.append("section").attr("class", "study-plot");
        const heading = panel.append("div").attr("class", "plot-heading");
        heading.append("h4"); heading.append("span").attr("class", "plot-scale");
        panel.append("p"); panel.append("svg");
        return panel;
      }).attr("data-backend", panel => panel.backend);
  });
  d3.select("#plots").selectAll(".study-plot").each(function(info, index) {
    const candidates = info.rows;
    d3.select(this).classed("unmeasured", !candidates.length).select("svg").attr("hidden", candidates.length ? null : "");
    if (!candidates.length) {
      d3.select(this).select("svg").selectAll("*").remove();
      d3.select(this).select("h4").text(info.backend.toUpperCase());
      d3.select(this).select(".plot-scale").text("Not measured");
      d3.select(this).select("p").text(`No ${info.backend.toUpperCase()} measurements for ${info.machine} in this selection.`);
      return;
    }
    const panel = this, first = candidates[0], values = candidates.filter(plottable);
    const unavailable = candidates.length - values.length;
    const counts = unique(candidates.map(row => row.count)).sort((a, b) => a - b).join("–");
    const workers = first.config.backend === "cpu"
      ? ` · ${unique(candidates.map(row => row.detail.worker_threads)).join("/")} compression threads` : "";
    d3.select(panel).select("h4").text(`${first.config.backend.toUpperCase()}${workers}`);
    d3.select(panel).select(".plot-scale").text(scaleNote);
    d3.select(panel).select("p").text(`${candidates.length} settings · ${counts} observations per setting${candidates.some(row => row.reference.drift) ? " · references varied" : ""}${unavailable ? ` · ${unavailable} unavailable on these axes` : ""}`);
    const scope = state.axes === "panel" ? values : matched.candidates;
    const boundaryScope = scope.filter(row => matched.ids.has(row.id) || resampledFrontier(row));
    const domains = plotDomains(state.extent === "frontier" && boundaryScope.length ? boundaryScope : scope);
    const inView = row => row.compression_fold >= domains.fold[0] && row.compression_fold <= domains.fold[1]
      && row.throughput.median >= domains.throughput[0] && row.throughput.median <= domains.throughput[1];
    for (const row of values) if (!inView(row)) outsideView.add(row.id);
    const width = Math.floor(panel.clientWidth - 2 * parseFloat(getComputedStyle(panel).paddingLeft));
    const height = width < 350 ? 270 : 290;
    const margin = {left: 56, top: 18, width: width - 74, height: height - 76};
    const x = d3.scaleLog().domain(domains.fold).range([0, margin.width]);
    const y = d3.scaleLinear().domain(domains.throughput).range([margin.height, 0]);
    const svg = d3.select(panel).select("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "group")
      .attr("aria-label", `${first.machine}, ${state.input === "all" ? "All datasets" : first.input_label}, ${first.config.backend}, ${first.config.sink}: logical compression fold on the logarithmic horizontal axis, logical throughput on the vertical axis. ${scaleNote}.`);
    svg.selectAll("*").remove();
    const plot = plotAxes(svg, x, y, {...margin,
      xLabel: "Logical compression fold (×)", yLabel: "Logical throughput (GiB/s)",
      xTicks: domains.fold[1] / domains.fold[0] < 10 ? d3.ticks(...domains.fold, width < 350 ? 3 : 5) : null});
    plot.selectAll(".plot-axis .tick line").attr("x2", function() { return this.getAttribute("x2") ? -4 : null; })
      .attr("y2", function() { return this.getAttribute("y2") ? 4 : null; });
    plot.selectAll(".plot-axis .tick text").attr("x", function() { return this.getAttribute("x") ? -10 : null; })
      .attr("y", function() { return this.getAttribute("y") ? 10 : null; });
    const clipId = `microscopy-plot-${index}`;
    svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
      .attr("width", margin.width).attr("height", margin.height);
    const points = plot.append("g").attr("clip-path", `url(#${clipId})`);
    if (!values.length) {
      plot.append("text").attr("class", "plot-empty").attr("x", margin.width / 2).attr("y", margin.height / 2)
        .attr("text-anchor", "middle").text("No values available on these axes");
      return;
    }
    if (domains.fold[0] <= 1 && domains.fold[1] >= 1) {
      plot.append("line").attr("class", "fold-baseline").attr("x1", x(1)).attr("x2", x(1))
        .attr("y1", 0).attr("y2", margin.height).append("title").text("1×: output equals logical input size");
    }
    const boundaries = [...d3.group(values.filter(row => result.ids.has(row.id)), row => row.condition).values()]
      .map(group => group.sort((a, b) => a.compression_fold - b.compression_fold));
    points.selectAll(".frontier-line").data(boundaries).join("path").attr("class", "frontier-line")
      .attr("data-condition", group => group[0].condition).attr("data-input", group => group[0].config.input_id)
      .style("stroke", group => state.input === "all" ? pointColor(group[0]) : "var(--text-secondary)")
      .attr("d", d3.line().x(row => x(row.compression_fold)).y(row => y(row.throughput.median)))
      .append("title").text(group => `${group[0].input_label}: observed frontier`);
    points.selectAll(".range").data(values.filter(row => row.count > 1
      && Number.isFinite(row.throughput.min) && Number.isFinite(row.throughput.max)), row => row.id)
      .join("line").attr("class", "range").attr("data-input", row => row.config.input_id)
      .attr("x1", row => x(row.compression_fold)).attr("x2", row => x(row.compression_fold))
      .attr("y1", row => y(row.throughput.min)).attr("y2", row => y(row.throughput.max))
      .attr("stroke", pointColor).attr("opacity", row => result.ids.has(row.id) || resampledFrontier(row) ? 0.75 : 0.25);
    points.selectAll(".fold-range").data(values.filter(row => row.count > 1 && row.compression_range), row => row.id)
      .join("line").attr("class", "fold-range").attr("data-input", row => row.config.input_id)
      .attr("x1", row => x(row.compression_range.min)).attr("x2", row => x(row.compression_range.max))
      .attr("y1", row => y(row.throughput.median)).attr("y2", row => y(row.throughput.median))
      .attr("stroke", pointColor).attr("opacity", 0.6);
    const marks = points.selectAll(".point")
      .data([...values].sort((a, b) => Number(result.ids.has(a.id)) - Number(result.ids.has(b.id))), row => row.id)
      .join("g").attr("class", row => result.ids.has(row.id) ? "point frontier" : resampledFrontier(row) ? "point resampled-frontier" : "point")
      .attr("data-id", row => row.id).attr("data-input", row => row.config.input_id).attr("tabindex", row => inView(row) ? 0 : -1).attr("role", "button")
      .attr("transform", row => `translate(${x(row.compression_fold)},${y(row.throughput.median)})`)
      .attr("aria-label", pointDescription)
      .on("click", (_, row) => selected(row.id))
      .on("keydown", (event, row) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selected(row.id); } });
    marks.append("circle").attr("r", matchMedia("(pointer: coarse)").matches ? 18 : 12).attr("fill", "transparent");
    marks.filter(row => !result.ids.has(row.id) && resampledFrontier(row)).append("circle")
      .attr("class", "frontier-ring").attr("r", 7).attr("fill", "none")
      .attr("stroke", pointColor).attr("stroke-width", 1.5);
    marks.append("path").attr("class", "mark")
      .attr("d", row => d3.symbol().type(pointShape(row.config.codec)).size(result.ids.has(row.id) ? 80 : 34)())
      .attr("fill", row => pointFill(row.config.codec, pointColor(row))).attr("stroke", pointColor)
      .attr("stroke-width", 1.4).attr("opacity", row => result.ids.has(row.id) ? 1 : resampledFrontier(row) ? 0.75 : 0.5);
    marks.append("title").text(pointDescription);
  });
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
    setting.append(button, element("small", `${row.machine} · ${row.input_label}`)); tr.dataset.id = row.id;
    tr.append(setting, element("td", `${row.config.backend.toUpperCase()} / ${sinkName(row.config.sink)}`),
      element("td", size(chunkBytes(row))), element("td", size(row.config.blosc_block_bytes)),
      element("td", format(row.throughput.median)), element("td", format(row.compression_fold)),
      element("td", evidence(row), row.reference.drift ? "drift" : result.ids.has(row.id) ? "frontier-label" : null),
      element("td", runDates(row.samples.map(sample => sample.started)), "run-date"));
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
  if (!row) { target.textContent = "Select a point to inspect its settings, observed ranges, and pipeline stages."; return; }
  const data = studies.get(row.study_id), raw = row.detail;
  if (!result.candidates.some(item => item.id === row.id)) target.append(element("p", "This configuration is outside the current filters."));
  else if (outsideView.has(row.id)) target.append(element("p", "This configuration is outside the frontier view. Choose All points to see its marker."));
  target.append(element("h3", codecName(row.config.codec)),
    element("p", `${row.input_label} · ${row.config.backend.toUpperCase()} · ${sinkName(row.config.sink)}`),
    pairs([
      ["Chunk", `${size(chunkBytes(row))} · ${raw.image_replay.chunk_shape.join(" × ")}`],
      ["Blosc block request", isBlosc(row) ? size(row.config.blosc_block_bytes) : "Not used"],
      ["Codec level", `${row.config.level}${raw.image_replay.codec_level_is_hint ? " (hint)" : ""}`],
      ["Spatial shard files", spatialShards(raw.image_replay)],
      ["Logical throughput", `${format(row.throughput.median)} GiB/s`],
      ["Observed range", `${format(row.throughput.min)}–${format(row.throughput.max)} GiB/s`],
      ["Logical compression fold", `${format(row.compression_fold)}×`],
      ["Spatial padding", `${format(row.padding_percent)}%`], ["Observations", row.count],
    ]));
  if (row.count === 1) target.append(element("p", "One observation cannot characterize run variation."));
  if (row.reference.drift) target.append(element("p", "Reference throughput varied across this session. These ranges may not describe variation between sessions."));
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
  target.append(element("h3", "Pipeline stages"), element("p", "Rates describe one recorded execution and use each stage’s own bytes. Intervals overlap and do not sum to elapsed time."));
  const scroll = element("div", null, "table-scroll"), table = element("table"), head = element("thead"), header = element("tr"), body = element("tbody");
  for (const title of ["Stage", "Avg ms", "In GiB/s", "Out GiB/s"]) { const th = element("th", title); th.scope = "col"; header.append(th); }
  head.append(header);
  for (const [name, stage] of Object.entries(raw.stages ?? {})) {
    const tr = element("tr"); tr.append(element("td", name), element("td", format(stage.avg_ms)),
      element("td", format(stage.in_gibs)), element("td", format(stage.out_gibs))); body.append(tr);
  }
  table.append(head, body); scroll.append(table); target.append(scroll);
  const conditions = element("details");
  conditions.append(element("summary", "Run conditions and reference variation"), pairs([
    ["Machine", data.study.machine.name], ["Data type", dtypeName(row.config.dtype)],
    ["Allowed CPU threads / compression workers", `${raw.execution_resources?.cpu_affinity?.length ?? data.study.machine.cpu_count} / ${raw.worker_threads}`],
    ...(data.study.machine.cpu_topology?.allowed_physical_cores != null
      ? [["Physical CPU cores", data.study.machine.cpu_topology.allowed_physical_cores]] : []),
    ["Output buffers / I/O workers", `${data.study.build.build_settings?.CHUCKY_OUTPUT_BUFFERS ?? "—"} / ${data.study.build.build_settings?.CHUCKY_IO_WORKERS ?? "—"}`],
    ["Chunks per shard", raw.image_replay.chunks_per_shard.join(" × ")],
    ["Measured window", `${format(raw.measurement.elapsed_s)} s`], ["Final drain", `${format(raw.measurement.drain_s)} s`],
    ["Nearby reference range", `${format(row.reference.spread_percent)}%`],
    ["All reference range", `${format(row.reference.condition_spread_percent)}%`],
  ]));
  target.append(conditions);
  const source = element("a", `Source ${data.study.build.revision.slice(0, 7)}`);
  source.href = `https://github.com/acquire-project/chucky/tree/${encodeURIComponent(data.study.build.revision)}`;
  const archive = element("a", "Retained raw observations"); archive.href = data.study.archive;
  const links = element("p"); links.append(source, document.createTextNode(" · "), archive);
  const details = element("details");
  details.append(element("summary", "Source and replay details"), links, element("pre", JSON.stringify({
    id: row.id, study: row.study_id, executions: row.samples, detail_execution: row.detail_execution,
    references: row.reference, binary_sha256: data.study.build.executable_sha256,
    image_input: raw.image_input, replay: raw.image_replay, command: raw.command,
    measurement: raw.measurement}, null, 2)));
  target.append(details);
}

function populateEntropy() {
  const inputs = unique(rows.map(row => row.config.input_id)).map(id => rows.find(row => row.config.input_id === id));
  $("entropy-body").replaceChildren(...inputs.map(row => {
    const sample = entropies.get(inputKey(row)), tr = element("tr"), label = element("th", row.input_label);
    label.scope = "row"; tr.dataset.input = row.config.input_id;
    const value = element("td", sample ? entropyValue(sample) : "Not sampled", "entropy-value");
    if (sample && entropySampleLimited(sample)) value.append(element("span", "sample limited", "entropy-limit"));
    tr.append(label, element("td", dtypeName(row.config.dtype)), value,
      element("td", sample ? format(sample.unique_values) : "—", "entropy-unique"),
      element("td", sample ? format(sample.sample_pixels) : "—", "entropy-pixels"));
    return tr;
  }));
  $("dataset-entropy").hidden = !inputs.some(row => entropies.has(inputKey(row)));
}

function renderEntropy() {
  const row = rows.find(row => row.config.input_id === state.input), sample = row && entropies.get(inputKey(row));
  $("entropy-summary").textContent = state.input === "all" ? "Sampled pixel entropy · compare datasets"
    : `Sampled pixel entropy: ${sample ? entropyValue(sample) + " bits/pixel"
      + (entropySampleLimited(sample) ? " · sample limited" : "") : "not available"}`;
  for (const tr of $("entropy-body").children) tr.classList.toggle("selected", tr.dataset.input === state.input);
}

function renderPreviewSources() {
  const sourceRows = unique(rows.map(row => row.config.input_id)).map(id => rows.find(row => row.config.input_id === id));
  $("preview-sources").replaceChildren(...sourceRows.flatMap(row => {
    const preview = previewFor(row);
    if (!preview) return [];
    const figure = element("figure"), caption = element("figcaption"), source = preview.source;
    const collection = element("a", source.collection), license = element("a", source.license);
    collection.href = source.url; license.href = source.license_url;
    const links = element("p"); links.append(collection, document.createTextNode(" · "), license);
    caption.append(element("h3", row.input_label),
      element("p", `${preview.modality.replaceAll("-", " ")} · ${preview.dtype} · ${preview.shape.join(" × ")} pixels [plane, y, x]`),
      element("p", source.attribution), links);
    figure.append(thumbnail(row), caption);
    return [figure];
  }));
  $("dataset-sources").hidden = !$("preview-sources").children.length;
}

function renderArchives() {
  $("archives").replaceChildren(element("h3", "Source data"));
  for (const data of studies.values()) {
    const selected = rows.filter(row => row.study_id === data.study.id);
    if (!selected.length) continue;
    const paragraph = element("p"), link = element("a", unique(selected.map(row => row.input_label)).join(", "));
    link.href = data.study.archive;
    paragraph.append(link, document.createTextNode(` · ${selected.length} displayed configurations · ${selected.reduce((sum, row) => sum + row.count, 0)} observations · source ${data.study.build.revision.slice(0, 7)}`));
    $("archives").append(paragraph);
  }
}

async function load() {
  $("retry").hidden = true;
  try {
    const index = await getJson("data/microscopy/index.json");
    if (index.version !== 1 || !Array.isArray(index.studies)) throw new Error("Unsupported study index");
    if (!index.studies.length) {
      $("load-status").textContent = "No microscopy measurements have been published yet.";
      return;
    }
    const sources = index.report ? new Set(index.report.flatMap(item => item.sources.map(source => source.study))) : null;
    const datasets = await Promise.all(index.studies.filter(item => !sources || sources.has(item.id)).map(item => getJson(item.file)));
    studies = new Map(datasets.map(data => [data.study.id, data]));
    previews = new Map((index.previews ?? []).map(preview => [`${preview.asset}:${preview.pack_sha256}`, preview]));
    entropies = new Map((index.entropy?.version === 2 ? index.entropy.inputs : [])
      .map(sample => [`${sample.asset}:${sample.pack_sha256}`, sample]));
    rows = reportRows(datasets, index.report);
    if (!rows.length) { $("load-status").textContent = "No microscopy measurements have been published yet."; return; }
    state = readState(location.search, rows);
    populateFilters(); populateEntropy(); sync(); renderArchives(); renderPreviewSources();
    $("load-status").hidden = true; $("workspace").hidden = false;
    render(); remember(true);
  } catch (error) {
    $("load-status").hidden = false;
    $("load-status").textContent = `Could not load microscopy measurements: ${error.message}`;
    $("retry").hidden = false;
  }
}

new ResizeObserver(entries => {
  document.documentElement.style.setProperty("--dataset-nav-height", `${entries[0].borderBoxSize?.[0]?.blockSize ?? entries[0].contentRect.height}px`);
}).observe($("dataset-navigation"));
$("retry").onclick = load;
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if ($("settings").open) { $("settings").open = false; $("settings").querySelector("summary").focus(); }
  else if (state?.selected && matchMedia("(max-width: 1100px)").matches) closeDetail();
});
window.addEventListener("popstate", () => { if (rows.length) { state = readState(location.search, rows); sync(); render(); } });
let resize;
window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => { if (result) { renderPlots(); highlight(); } }, 120); });
wireThemeToggle(() => { if (result) { renderPlots(); highlight(); } });
await load();
