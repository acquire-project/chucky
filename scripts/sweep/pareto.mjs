// The sole Pareto implementation, shared by the browser and Node tests.
// Values are used at their archived precision. Ties remain on the frontier.
export const isBlosc = row => row.control !== true && row.codec?.startsWith("blosc-");
export const memoryValue = row => row.estimated_device_gib;

export function eligible(rows, filters = {}) {
  const chosen = (values, value) => values == null || values.includes(value);
  return rows.filter(row =>
    chosen(filters.systems, row.experiment_id) &&
    (!filters.workload || filters.workload === "all" || row.workload_id === filters.workload) &&
    chosen(filters.codecs, row.codec.replace(/^blosc-/, "")) &&
    chosen(filters.shuffles, row.shuffle) && chosen(filters.blocks, row.block_kib) &&
    (filters.budget == null || Number.isFinite(row.estimated_device_gib) && row.estimated_device_gib <= filters.budget));
}

export function objectives(row) {
  const values = [row.throughput_gibs?.median, row.compression_fold];
  return values.every(Number.isFinite) ? values : null;
}

export function dominates(a, b) {
  return a != null && b != null && a.length === b.length &&
    a.every((value, i) => value >= b[i]) && a.some((value, i) => value > b[i]);
}

export function frontier(rows, {mode = "codec", ...filters} = {}) {
  if (!["codec", "cross"].includes(mode)) throw new Error(`Unknown frontier mode: ${mode}`);
  const candidates = eligible(rows, filters);
  const groups = new Map();
  for (const row of candidates) {
    if (!isBlosc(row)) continue;
    const values = objectives(row);
    if (!values) continue;
    // workload_id includes full geometry, data type, batch, frame count and sink.
    // Overlay changes presentation only; systems never compete for membership.
    const key = JSON.stringify([row.experiment_id, row.workload_id, mode === "codec" ? row.codec : null]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({row, values});
  }
  const ids = new Set();
  for (const group of groups.values()) {
    for (const point of group) {
      if (!group.some(other => dominates(other.values, point.values))) ids.add(point.row.id);
    }
  }
  return {candidates, ids};
}

export function defaultState(systems) {
  return {systems: [...systems], codecs: null, shuffles: null, blocks: null, budget: null,
    mode: "codec", view: "compression", layout: "matrix", workload: "all",
    extent: "full", selected: null, sort: "throughput", direction: "desc"};
}

export function readState(search, systems, workloads, blocks) {
  const p = new URLSearchParams(search), state = defaultState(systems);
  const array = (key, allowed, numeric = false) => {
    if (!p.has(key) || p.get(key) === "all") return null;
    return p.get(key).split(",").filter(Boolean).map(x => numeric ? Number(x) : x).filter(x => allowed.includes(x));
  };
  state.systems = array("systems", systems) ?? systems;
  state.codecs = array("codecs", ["lz4", "zstd"]);
  state.shuffles = array("shuffles", ["none", "byte", "bit"]);
  state.blocks = array("blocks", blocks, true);
  for (const [key, allowed] of Object.entries({mode: ["codec", "cross"],
    view: ["compression", "memory"], layout: ["matrix", "overlay"], extent: ["full", "fit"],
    workload: ["all", ...workloads], sort: ["system", "workload", "codec", "shuffle", "block", "throughput", "fold", "measured", "estimated", "repetitions"],
    direction: ["asc", "desc"]})) {
    if (allowed.includes(p.get(key))) state[key] = p.get(key);
  }
  if (p.has("budget") && p.get("budget").trim() && Number.isFinite(+p.get("budget")) && +p.get("budget") >= 0)
    state.budget = +p.get("budget");
  state.selected = p.get("selected") || null;
  return state;
}

export function writeState(state) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state)) {
    if (value != null) params.set(key, Array.isArray(value) ? value.join(",") : typeof value === "boolean" ? +value : value);
  }
  return params.toString();
}

export function measurementsCsv(rows, frontierIds, experiments = new Map()) {
  const columns = ["id", "experiment", "configuration_id", "workload_id", "fill", "chunk_kib", "codec", "shuffle", "block_kib", "level",
    "repetitions", "throughput_median_gibs", "throughput_min_gibs", "throughput_max_gibs", "compression_fold",
    "measured_device_median_gib", "measured_device_min_gib", "measured_device_max_gib", "estimated_device_gib", "estimated_pinned_gib",
    "frontier", "control", "summary", "summary_line", "raw", "source_metrics_json"];
  const cell = value => {
    let text = value == null ? "" : String(value);
    if (/^[=+@\t\r]/.test(text) || /^-[^\d.]/.test(text)) text = "'" + text;
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return columns.join(",") + "\r\n" + rows.map(r => [r.id, experiments.get(r.experiment_id)?.label ?? r.experiment_id,
    r.configuration_id, r.workload_id, r.fill, r.chunk_kib, r.codec, r.shuffle, r.block_kib, r.level, r.repetitions,
    r.throughput_gibs.median, r.throughput_gibs.min, r.throughput_gibs.max, r.compression_fold,
    r.measured_device_gib.median, r.measured_device_gib.min, r.measured_device_gib.max, r.estimated_device_gib, r.estimated_pinned_gib,
    frontierIds.has(r.id), r.control, r.provenance.summary, r.provenance.summary_line, r.provenance.raw,
    JSON.stringify(r.source_metrics)].map(cell).join(",")).join("\r\n") + (rows.length ? "\r\n" : "");
}
