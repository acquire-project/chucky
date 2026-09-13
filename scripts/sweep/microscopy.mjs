export const isBlosc = row => row.config.codec.startsWith("blosc-");
export const chunkBytes = row => {
  const match = /^(\d+)([KM])$/.exec(row.config.chunk_label);
  return match ? +match[1] * (match[2] === "M" ? 1048576 : 1024) : NaN;
};

export function eligible(rows, state = {}) {
  const matches = (key, value) => !state[key] || state[key] === "all" || state[key] === String(value);
  return rows.filter(row => matches("study", row.study_id) && matches("input", row.config.input_id)
    && matches("backend", row.config.backend) && matches("sink", row.config.sink)
    && matches("codec", row.config.codec.replace(/^blosc-/, ""))
    && matches("chunk", row.config.chunk_label)
    && (!isBlosc(row) || matches("block", row.config.blosc_block_bytes)));
}

export function dominates(a, b) {
  return a.throughput.median >= b.throughput.median && a.compression_fold >= b.compression_fold
    && (a.throughput.median > b.throughput.median || a.compression_fold > b.compression_fold);
}

export function frontier(rows, state = {}) {
  const candidates = eligible(rows, state), groups = new Map(), ids = new Set();
  for (const row of candidates) {
    if (row.reference.drift || !Number.isFinite(row.throughput.median) || row.throughput.median <= 0
        || !Number.isFinite(row.compression_fold) || row.compression_fold <= 0) continue;
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
    study: rows.map(row => row.study_id), input: rows.map(row => row.config.input_id),
    backend: rows.map(row => row.config.backend), sink: rows.map(row => row.config.sink),
    codec: rows.map(row => row.config.codec.replace(/^blosc-/, "")),
    chunk: rows.map(row => row.config.chunk_label),
    block: rows.filter(isBlosc).map(row => String(row.config.blosc_block_bytes)),
  };
  for (const [key, values] of Object.entries(choices)) {
    const value = params.get(key);
    state[key] = value === "all" || values.includes(value) ? value
      : key === "input" ? values[0] ?? "all" : "all";
  }
  state.selected = rows.some(row => row.id === params.get("selected")) ? params.get("selected") : null;
  return state;
}

export function writeState(state) {
  return new URLSearchParams(Object.entries(state).filter(([, value]) => value != null)).toString();
}

export function measurementsCsv(rows, frontierIds) {
  const columns = ["study", "id", "input", "backend", "sink", "codec", "shuffle", "level", "chunk_bytes",
    "block_bytes_requested", "logical_gibs_median", "logical_gibs_min", "logical_gibs_max", "logical_compression_fold",
    "padding_percent", "observations", "reference_spread_percent", "condition_reference_spread_percent", "reference_drift", "observed_frontier", "needs_confirmation"];
  const cell = value => {
    let text = value == null ? "" : String(value);
    if (/^[=+@\t\r]/.test(text) || /^-[^\d.]/.test(text)) text = "'" + text;
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return columns.join(",") + "\r\n" + rows.map(row => [row.study_id, row.id, row.config.input_id,
    row.config.backend, row.config.sink, row.config.codec, row.config.blosc_shuffle, row.config.level, chunkBytes(row),
    row.config.blosc_block_bytes, row.throughput.median, row.throughput.min, row.throughput.max, row.compression_fold,
    row.padding_percent, row.count, row.reference.spread_percent, row.reference.condition_spread_percent, row.reference.drift,
    frontierIds.has(row.id), row.needs_confirmation].map(cell).join(",")).join("\r\n") + "\r\n";
}
