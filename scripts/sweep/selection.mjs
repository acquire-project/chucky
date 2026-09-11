import {bloscBlockKey, bloscBlockLabel, matchesBloscBlock} from "./blosc.js";

export function metricValue(run, key) {
  const parts = key.split(".");
  let v = run;
  for (const p of parts) {
    if (v == null || typeof v !== "object") return null;
    v = v[p];
  }
  return typeof v === "number" && isFinite(v) ? v : null;
}

export function matchesSetup(run, state) {
  return (run.codec_label ?? run.codec) === state.codec && run.backend === state.backend && run.sink === state.sink
    && matchesBloscBlock(run, state.bloscBlock);
}

export function comparable(sweep, meta) {
  const retiredKey = meta.retired || meta.key.split(".").pop();
  return !(sweep.retired || []).includes(retiredKey);
}

export function bestRun(sweep, scenario, fill, state, meta) {
  const wantHigh = meta.better === "high";
  let best = null, bestValue = null, count = 0;
  for (const run of sweep.runs) {
    if (run.scenario !== scenario || run.fill !== fill) continue;
    if (run.status !== "pass" || !matchesSetup(run, state)) continue;
    const value = metricValue(run, state.metric);
    if (value == null) continue;
    count++;
    if (bestValue == null || (wantHigh ? value > bestValue : value < bestValue)) {
      best = run; bestValue = value;
    }
  }
  return best ? {run: best, value: bestValue, count} : null;
}

export function percentChange(previous, latest) {
  if (previous == null || latest == null || previous === 0) return null;
  return (latest - previous) / Math.abs(previous) * 100;
}

export function configLabel(run) {
  const block = bloscBlockKey(run);
  return [run.dtype, run.chunk_bytes_label, block ? `block ${bloscBlockLabel(block)}` : ""]
    .filter(Boolean).join(" · ");
}

export function moversFor(machine, state, meta) {
  const history = new Map();
  let newest = null;
  for (const sweep of machine.sweeps) {
    if (!comparable(sweep, meta)) continue;
    for (const run of sweep.runs) {
      if (run.status !== "pass" || !matchesSetup(run, state)) continue;
      const value = metricValue(run, state.metric);
      if (value == null) continue;
      if (!history.has(run.id)) history.set(run.id, []);
      history.get(run.id).push({sweep, run, value});
      newest = sweep;
    }
  }

  const rows = [];
  for (const seen of history.values()) {
    const now = seen[seen.length - 1];
    if (seen.length < 2 || now.sweep !== newest) continue;
    const then = seen[seen.length - 2];
    const pct = percentChange(then.value, now.value);
    if (pct == null) continue;
    rows.push({machine, run: now.run, previous: then.value, latest: now.value, pct,
               latestSweep: now.sweep, previousSweep: then.sweep});
  }
  return {rows, newest};
}

/** Explorer selection is independent of DOM controls and chart grouping. */
export function filterRuns(runs, selection, {includeBackend = true} = {}) {
  const {codec, fill, backend, dtype, sink, bloscBlock, scenarios, s3Throughput} = selection;
  return runs.filter(run => {
    if ((run.codec_label ?? run.codec) !== codec || run.fill !== fill || run.dtype !== dtype || run.sink !== sink) return false;
    if (!matchesBloscBlock(run, bloscBlock)) return false;
    if (includeBackend && run.backend !== backend) return false;
    if (!scenarios.has(run.scenario)) return false;
    if (sink === "s3" && s3Throughput && String(run.s3_throughput_gbps) !== s3Throughput) return false;
    return true;
  });
}
