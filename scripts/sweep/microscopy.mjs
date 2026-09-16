export const isBlosc = row => row.config.codec.startsWith("blosc-");
export const chunkBytes = row => {
  const match = /^(\d+)([KM])$/.exec(row.config.chunk_label);
  return match ? +match[1] * (match[2] === "M" ? 1048576 : 1024) : NaN;
};

export function reportRows(datasets, report) {
  const studies = new Map(datasets.map(data => [data.study.id, data]));
  if (!report) return datasets.flatMap(data => data.measurements.map(row => ({...row, machine: data.study.machine.name})));
  const selected = report.flatMap(item => item.sources.flatMap(source => {
    const data = studies.get(source.study);
    if (!data) throw new Error(`Missing report source: ${source.study}`);
    const rows = data.measurements.filter(row => row.config.input_id === item.input && source.backends.includes(row.config.backend));
    if (source.backends.some(backend => !rows.some(row => row.config.backend === backend)))
      throw new Error(`Missing report measurements: ${item.input}`);
    return rows.map(row => ({...row, input_label: item.label, machine: data.study.machine.name}));
  }));
  if (new Set(selected.map(row => row.id)).size !== selected.length) throw new Error("Duplicate report measurements");
  return selected;
}

export function eligible(rows, state = {}) {
  const matches = (key, value) => !state[key] || state[key] === "all" || state[key] === String(value);
  return rows.filter(row => matches("machine", row.machine) && matches("input", row.config.input_id)
    && matches("backend", row.config.backend) && matches("sink", row.config.sink)
    && matches("codec", row.config.codec)
    && matches("chunk", row.config.chunk_label)
    && (!isBlosc(row) || matches("block", row.config.blosc_block_bytes)));
}

export function dominates(a, b) {
  return a.throughput.median >= b.throughput.median && a.compression_fold >= b.compression_fold
    && (a.throughput.median > b.throughput.median || a.compression_fold > b.compression_fold);
}

export const resampledFrontier = row => Number.isFinite(row.uncertainty?.frontier_frequency)
  && row.uncertainty.frontier_frequency >= 0.05;

export const plottable = row => Number.isFinite(row.compression_fold) && row.compression_fold > 0
  && Number.isFinite(row.throughput.median) && row.throughput.median > 0;

export function plotDomains(rows) {
  const points = rows.filter(plottable);
  if (!points.length) return {fold: [0.5, 2], throughput: [0, 1]};
  const folds = points.flatMap(row => [row.compression_fold, row.compression_range?.min, row.compression_range?.max])
    .filter(value => Number.isFinite(value) && value > 0);
  const rates = points.flatMap(row => [row.throughput.min, row.throughput.median, row.throughput.max])
    .filter(value => Number.isFinite(value) && value >= 0);
  const loFold = Math.min(...folds), hiFold = Math.max(...folds);
  const loRate = Math.min(...rates), hiRate = Math.max(...rates);
  const foldPad = Math.max(1.025, (hiFold / loFold) ** 0.06);
  const ratePad = Math.max((hiRate - loRate) * 0.06, hiRate * 0.015, 0.001);
  return {fold: [loFold / foldPad, hiFold * foldPad],
    throughput: [Math.max(0, loRate - ratePad), hiRate + ratePad]};
}

export function frontier(rows, state = {}) {
  const candidates = eligible(rows, state), groups = new Map(), ids = new Set();
  for (const row of candidates) {
    if ((row.reference.drift && row.phase !== "comparison") || !plottable(row)) continue;
    if (!groups.has(row.condition)) groups.set(row.condition, []);
    groups.get(row.condition).push(row);
  }
  for (const group of groups.values()) {
    for (const row of group) if (!group.some(other => dominates(other, row))) ids.add(row.id);
  }
  return {candidates, ids};
}

export function readState(search, rows) {
  const params = new URLSearchParams(search), state = {};
  const choices = {
    machine: rows.map(row => row.machine), input: rows.map(row => row.config.input_id),
    backend: rows.map(row => row.config.backend), sink: rows.map(row => row.config.sink),
    codec: rows.map(row => row.config.codec),
    chunk: rows.map(row => row.config.chunk_label),
    block: rows.filter(isBlosc).map(row => String(row.config.blosc_block_bytes)),
  };
  for (const [key, values] of Object.entries(choices)) {
    const value = params.get(key);
    state[key] = value === "all" || values.includes(value) ? value
      : key === "input" ? values[0] ?? "all" : "all";
  }
  state.axes = ["panel", "input", "all"].includes(params.get("axes")) ? params.get("axes") : "panel";
  state.extent = !params.has("extent") || params.get("extent") === "frontier" ? "frontier" : "all";
  state.selected = rows.some(row => row.id === params.get("selected")) ? params.get("selected") : null;
  return state;
}

export function writeState(state) {
  return new URLSearchParams(Object.entries(state).filter(([, value]) => value != null)).toString();
}

export function measurementsCsv(rows, frontierIds) {
  const columns = ["study", "id", "input", "backend", "sink", "codec", "shuffle", "level", "chunk_bytes",
    "block_bytes_requested", "logical_gibs_median", "logical_gibs_min", "logical_gibs_max", "logical_compression_fold",
    "padding_percent", "observations", "reference_spread_percent", "condition_reference_spread_percent", "reference_drift", "observed_frontier", "needs_confirmation", "logical_fold_min", "logical_fold_max", "resampled_frontier_frequency",
    "bootstrap_throughput_lower", "bootstrap_throughput_upper", "bootstrap_fold_lower", "bootstrap_fold_upper", "resampling_scope"];
  const cell = value => {
    let text = value == null ? "" : String(value);
    if (/^[=+@\t\r]/.test(text) || /^-[^\d.]/.test(text)) text = "'" + text;
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return columns.join(",") + "\r\n" + rows.map(row => [row.study_id, row.id, row.config.input_id,
    row.config.backend, row.config.sink, row.config.codec, row.config.blosc_shuffle, row.config.level, chunkBytes(row),
    row.config.blosc_block_bytes, row.throughput.median, row.throughput.min, row.throughput.max, row.compression_fold,
    row.padding_percent, row.count, row.reference.spread_percent, row.reference.condition_spread_percent, row.reference.drift,
    frontierIds.has(row.id), row.needs_confirmation, row.compression_range?.min, row.compression_range?.max,
    row.uncertainty?.frontier_frequency, row.uncertainty?.throughput.lower, row.uncertainty?.throughput.upper,
    row.uncertainty?.compression_fold.lower, row.uncertainty?.compression_fold.upper, row.uncertainty?.scope].map(cell).join(",")).join("\r\n") + "\r\n";
}
