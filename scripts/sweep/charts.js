// Small presentation helpers shared by report charts. D3 is a pinned local
// bundle loaded before page modules. Analysis lives in pure modules instead.
export function fmtTick(value) {
  return Math.abs(value) >= 1000 ? d3.format("~s")(value) : d3.format("~f")(value);
}

export function fmt(value) {
  if (value == null) return "—";
  const abs = Math.abs(value);
  if (abs >= 1000) return d3.format(",.0f")(value);
  if (abs >= 100) return d3.format(".0f")(value);
  if (abs >= 10) return d3.format(".1f")(value);
  if (abs >= 1) return d3.format(".2f")(value);
  if (abs > 0) return d3.format(".3f")(value);
  return "0";
}

export function fmtSignificant(value, digits = 3) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toPrecision(digits);
}

export function plotAxes(svg, x, y, {left, top, width, height, xLabel, yLabel}) {
  const g = svg.append("g").attr("transform", `translate(${left},${top})`);
  g.append("g").attr("class", "plot-grid").call(d3.axisLeft(y).ticks(4).tickSize(-width).tickFormat(""));
  g.append("g").attr("class", "plot-axis").attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(4).tickFormat(typeof x.base === "function" ? x.tickFormat(4, fmtTick) : fmtTick).tickSizeOuter(0));
  g.append("g").attr("class", "plot-axis").call(d3.axisLeft(y).ticks(4).tickSizeOuter(0).tickFormat(fmtTick));
  svg.append("text").attr("class", "axis-title").attr("x", left + width / 2)
    .attr("y", top + height + 42).attr("text-anchor", "middle").text(xLabel);
  svg.append("text").attr("class", "axis-title").attr("transform", `translate(15,${top + height / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").text(yLabel);
  return g;
}
