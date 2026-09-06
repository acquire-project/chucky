import {memoryValue} from "./pareto.mjs";
import {plotAxes, fmtSignificant} from "./charts.js";

const makeElement = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};

export const formatSize = kib => kib === 0 ? "Raw" : kib >= 1024 ? `${kib / 1024} MiB` : `${kib} KiB`;

const inputName = fill => fill === "rand" ? "12-bit random" : fill === "xor" ? "Coordinate XOR" : fill;

export const workloadName = workload => `${inputName(workload.fill)} / ${formatSize(workload.chunk_kib)} chunks`;

const pointShape = row => row.control ? d3.symbolDiamond : ({
  none: d3.symbolSquare,
  byte: d3.symbolCircle,
  bit: d3.symbolTriangle,
}[row.shuffle] ?? d3.symbolCross);

const codecColor = row => row.codec.endsWith("lz4") ? "var(--codec-lz4)" : "var(--codec-zstd)";
const systemDash = index => [null, "7 4", "2 4", "8 3 2 3"][index % 4];

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

function linearDomain(values, fit) {
  if (!values.length) return [0, 1];
  const lo = d3.min(values), hi = d3.max(values);
  const pad = Math.max((hi - lo) * .06, Math.abs(hi) * .015, .001);
  return [fit ? Math.max(0, lo - pad) : 0, hi + pad];
}

function logDomain(values, fit) {
  if (!values.length) return [1, 10];
  const lo = d3.min(values), hi = d3.max(values);
  const pad = Math.max(1.025, Math.pow(hi / lo, .06));
  // One is the meaningful uncompressed baseline. Fitted views instead use a
  // multiplicative pad so narrow random-data ranges remain readable.
  return [fit ? lo / pad : Math.min(1, lo / pad), hi * pad];
}

function legendMark(label, row, {faint = false, line = false, lineIndex = 0} = {}) {
  const node = makeElement("span", "legend-item");
  const svg = d3.create("svg").attr("viewBox", "-11 -11 22 22").attr("aria-hidden", "true");
  if (line) {
    svg.append("path").attr("d", "M-11,0H11").attr("stroke", "var(--text-secondary)")
      .attr("stroke-width", 2).attr("stroke-dasharray", systemDash(lineIndex));
    svg.append("circle").attr("r", 5).attr("stroke", "var(--text-secondary)").attr("stroke-width", 1.5)
      .attr("fill", systemFill(svg, `legend-system-${lineIndex}`, lineIndex, "var(--text-secondary)"));
  } else {
    svg.append("path").attr("d", d3.symbol().type(pointShape(row)).size(65)())
      .attr("fill", row.control ? "var(--surface-1)" : row.codec ? codecColor(row) : "var(--text-secondary)")
      .attr("stroke", row.codec ? codecColor(row) : "var(--text-secondary)").attr("opacity", faint ? .3 : 1);
  }
  node.append(svg.node(), document.createTextNode(label));
  return node;
}

/** Owns the D3 plot matrix and its selection-only redraws. */
export function createPlots({container, legend, experiments, workloads, onSelect}) {
  let state, result;
  let selectionDraws = [];

  const xValue = row => state.view === "memory" ? memoryValue(row) : row.compression_fold;

  function renderLegend() {
    legend.replaceChildren(
      legendMark("LZ4", {codec: "lz4", shuffle: "byte"}),
      legendMark("Zstd", {codec: "zstd", shuffle: "byte"}),
      legendMark("No shuffle", {shuffle: "none"}),
      legendMark("Byte shuffle", {shuffle: "byte"}),
      legendMark("Bitshuffle", {shuffle: "bit"}),
      legendMark("Other candidates", {shuffle: "byte"}, {faint: true}),
      legendMark("Raw control (excluded)", {control: true}),
    );
    if (state.layout === "overlay") {
      state.systems.forEach((id, i) => legend.append(legendMark(experiments.get(id).label, {}, {line: true, lineIndex: i})));
    }
  }

  function drawChart(cell, points, xDomain, yDomain, workload) {
    const width = Math.max(280, Math.round(cell.getBoundingClientRect().width)), height = 280;
    const margin = {left: 52, top: 14, width: width - 72, height: height - 68};
    const svg = d3.select(cell).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
      .attr("tabindex", points.length ? 0 : -1).attr("role", "group")
      .attr("aria-label", `${cell.querySelector("h4").textContent}, ${workloadName(workload)}. ${points.length} points. Arrow keys select, Home and End jump.`);
    const x = d3.scaleLog().domain(xDomain).range([0, margin.width]);
    const y = d3.scaleLinear().domain(yDomain).range([margin.height, 0]);
    const xLabel = state.view === "compression" ? "Reported compression fold (×)" : "Estimated device allocation (GiB)";
    const plot = plotAxes(svg, x, y, {...margin, xLabel, yLabel: "Input throughput (GiB/s)"});
    if (!points.length) {
      plot.append("text").attr("class", "plot-empty").attr("x", margin.width / 2).attr("y", margin.height / 2)
        .attr("text-anchor", "middle").text("No eligible measurements");
      return;
    }

    const order = [...points].sort((a, b) => xValue(a) - xValue(b) || b.throughput_gibs.median - a.throughput_gibs.median || a.id.localeCompare(b.id, "en"));
    // Lines connect tested frontier settings as a reading aid; they do not
    // interpolate performance between settings.
    const frontiers = d3.group(order.filter(row => result.ids.has(row.id)),
      row => `${row.experiment_id}:${state.mode === "codec" ? row.codec : "all"}`);
    for (const group of frontiers.values()) {
      plot.append("path").attr("fill", "none")
        .attr("stroke", state.mode === "codec" ? codecColor(group[0]) : "var(--text-secondary)")
        .attr("stroke-width", 1.4).attr("opacity", .65)
        .attr("stroke-dasharray", state.layout === "overlay" ? systemDash(state.systems.indexOf(group[0].experiment_id)) : null)
        .attr("d", d3.line().x(row => x(xValue(row))).y(row => y(row.throughput_gibs.median))(group));
    }

    const marks = plot.append("g").selectAll("g")
      .data([...order].sort((a, b) => Number(result.ids.has(a.id)) - Number(result.ids.has(b.id))))
      .join("g").attr("class", "chart-point").attr("data-id", row => row.id)
      .attr("transform", row => `translate(${x(xValue(row))},${y(row.throughput_gibs.median)})`)
      .on("click", (event, row) => { svg.node().focus({preventScroll: true}); onSelect(row.id); });
    marks.append("circle").attr("r", 10).attr("fill", "transparent");
    marks.append("path").attr("class", "mark")
      .attr("d", row => d3.symbol().type(pointShape(row)).size(result.ids.has(row.id) ? 70 : 30)())
      .attr("fill", row => row.control ? "var(--surface-1)" : state.layout !== "overlay" ? codecColor(row)
        : systemFill(svg, `fill-${workload.id}-${row.experiment_id}-${row.codec}`, state.systems.indexOf(row.experiment_id), codecColor(row)))
      .attr("stroke", codecColor).attr("stroke-width", 1.25)
      .attr("opacity", row => result.ids.has(row.id) || row.control ? 1 : .28);
    marks.append("title").text(row => `${experiments.get(row.experiment_id).label} · ${row.codec} / ${row.shuffle} / ${formatSize(row.block_kib)}\n${fmtSignificant(row.throughput_gibs.median)} GiB/s (${fmtSignificant(row.throughput_gibs.min)}–${fmtSignificant(row.throughput_gibs.max)}); ${fmtSignificant(row.compression_fold)}×`);

    const range = plot.append("g").attr("class", "range");
    const drawSelection = () => {
      marks.classed("selected", row => row.id === state.selected);
      range.selectAll("*").remove();
      const row = points.find(point => point.id === state.selected);
      if (!row) return;
      const px = x(xValue(row)), lo = y(row.throughput_gibs.min), hi = y(row.throughput_gibs.max);
      range.append("path").attr("d", `M${px},${lo}V${hi}M${px - 4},${lo}H${px + 4}M${px - 4},${hi}H${px + 4}`);
    };
    selectionDraws.push(drawSelection);
    drawSelection();

    svg.on("focus", () => { if (!points.some(row => row.id === state.selected)) onSelect(order[0].id, false); });
    svg.on("keydown", event => {
      const current = order.findIndex(row => row.id === state.selected);
      let next;
      if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (current + 1) % order.length;
      else if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (current - 1 + order.length) % order.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = order.length - 1;
      else return;
      event.preventDefault();
      onSelect(order[next].id, false);
    });
  }

  function renderCharts() {
    container.replaceChildren();
    selectionDraws = [];
    if (!state.systems.length) return;
    for (const workload of workloads.values()) {
      if (state.workload !== "all" && workload.id !== state.workload) continue;
      const candidates = result.candidates.filter(row => row.workload_id === workload.id);
      const plotted = candidates.filter(row => Number.isFinite(xValue(row)) && xValue(row) > 0);
      const section = makeElement("section", "workload-row");
      const title = makeElement("h3", null, workloadName(workload));
      title.append(makeElement("small", null, `Shape ${workload.shape.join(" × ")} · ${workload.padded_batch_bytes / 2**20} MiB batch`));
      section.append(title);
      const matrix = makeElement("div", "matrix-row");
      matrix.tabIndex = 0;
      matrix.setAttribute("role", "region");
      matrix.setAttribute("aria-label", `${workloadName(workload)} systems`);
      matrix.style.setProperty("--systems", state.layout === "overlay" ? 1 : state.systems.length);
      section.append(matrix);
      container.append(section);

      const xDomain = logDomain(plotted.map(xValue), state.extent === "fit");
      const yDomain = linearDomain(plotted.flatMap(row => [row.throughput_gibs.min, row.throughput_gibs.max]), state.extent === "fit");
      const groups = state.layout === "overlay" ? [state.systems] : state.systems.map(id => [id]);
      for (const systemIds of groups) {
        const cell = makeElement("div", "plot-cell");
        const points = plotted.filter(row => systemIds.includes(row.experiment_id));
        cell.append(makeElement("h4", null, systemIds.map(id => experiments.get(id).label).join(" / ")));
        const missing = candidates.filter(row => systemIds.includes(row.experiment_id)).length - points.length;
        cell.append(makeElement("p", "plot-meta", `${points.length} candidates${missing ? ` · ${missing} unavailable on this axis` : ""}`));
        matrix.append(cell);
        drawChart(cell, points, xDomain, yDomain, workload);
      }
    }
  }

  return {
    render(nextState, nextResult) {
      state = nextState;
      result = nextResult;
      renderLegend();
      renderCharts();
    },
    redraw() {
      if (state && result) {
        renderLegend();
        renderCharts();
      }
    },
    updateSelection() {
      selectionDraws.forEach(draw => draw());
    },
  };
}
