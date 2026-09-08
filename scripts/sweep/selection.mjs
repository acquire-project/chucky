import {bloscBlockKey, bloscBlockLabel} from "./blosc.js";

export function performanceSweeps(sweeps) {
  return sweeps.filter(sweep => !sweep.smoke);
}

export function inputKey(run) {
  return run.input_id || run.fill;
}

export function inputLabel(run) {
  return run.input_label || run.fill;
}

export function inputLabels(runs) {
  const labels = new Map(runs.map(run => [inputKey(run), inputLabel(run)]));
  const counts = new Map();
  for (const label of labels.values()) counts.set(label, (counts.get(label) || 0) + 1);
  for (const [key, label] of labels) {
    if (counts.get(label) > 1) labels.set(key, label + " [" + key.slice(-8) + "]");
  }
  return labels;
}

/** Recorded codec settings for point tooltips, without changing its selector key. */
export function codecSettings(run) {
  const settings = [];
  if (run.blosc_shuffle != null) settings.push(`${run.blosc_shuffle} shuffle`);
  const level = run.codec?.startsWith("blosc-") ? run.blosc_level : run.level;
  if (level != null) settings.push(`level ${level}`);
  return settings.join(", ");
}

/** Keep registry order, but expose only IDs observed in this report scope. */
export function scopeWorkloads(workloads, runs) {
  const observedInputs = new Set(runs.map(inputKey));
  const observedScenarios = new Set(runs.map(run => run.scenario));
  return {
    version: workloads.version,
    inputs: workloads.inputs
      .filter(input => observedInputs.has(input.id))
      .map(input => ({...input,
        scenarios: input.scenarios.filter(id => observedScenarios.has(id))})),
    scenarios: workloads.scenarios
      .filter(scenario => observedScenarios.has(scenario.id))
      .map(scenario => ({...scenario,
        inputs: scenario.inputs.filter(id => observedInputs.has(id))})),
  };
}

function entry(entries, id) {
  return entries.find(candidate => candidate.id === id) || null;
}

export function workloadsCompatible(workloads, scenario, input) {
  return entry(workloads.scenarios, scenario)?.inputs.includes(input) || false;
}

export function preferredScenario(workloads, input) {
  const selected = entry(workloads.inputs, input);
  if (!selected) return null;
  return selected.scenarios.includes(selected.default_scenario)
    ? selected.default_scenario : selected.scenarios[0] || null;
}

export function preferredInput(workloads, scenario) {
  const selected = entry(workloads.scenarios, scenario);
  if (!selected) return null;
  return selected.inputs.includes(selected.default_input)
    ? selected.default_input : selected.inputs[0] || null;
}

/** Reconcile the overview's single Scenario control after Input changes. */
export function reconcileInput(workloads, selection, input) {
  const scenario = workloadsCompatible(workloads, selection.scenario, input)
    ? selection.scenario : preferredScenario(workloads, input);
  return {scenario, input};
}

/** Reconcile the overview's single Input control after Scenario changes. */
export function reconcileScenario(workloads, selection, scenario) {
  const input = workloadsCompatible(workloads, scenario, selection.input)
    ? selection.input : preferredInput(workloads, scenario);
  return {scenario, input};
}

/** Retain only compatible checks after an Explorer Input change. */
export function reconcileScenarioSet(workloads, scenarios, input) {
  const kept = new Set(
    [...scenarios].filter(scenario =>
      workloadsCompatible(workloads, scenario, input))
  );
  if (!kept.size) {
    const fallback = preferredScenario(workloads, input);
    if (fallback) kept.add(fallback);
  }
  return kept;
}

/** Check one Explorer group, excluding its scenarios incompatible with Input. */
export function selectScenarioGroup(workloads, scenarios, group, input) {
  const selected = new Set(scenarios);
  for (const scenario of group) {
    if (workloadsCompatible(workloads, scenario, input)) selected.add(scenario);
    else selected.delete(scenario);
  }
  return selected;
}

/** Apply one explicit Explorer scenario check or uncheck. */
export function reconcileScenarioToggle(workloads, selection, scenario, checked) {
  const scenarios = new Set(selection.scenarios);
  if (!checked) {
    scenarios.delete(scenario);
    return {input: selection.input, scenarios};
  }
  scenarios.add(scenario);
  if (workloadsCompatible(workloads, scenario, selection.input)) {
    return {input: selection.input, scenarios};
  }
  const input = preferredInput(workloads, scenario);
  return {input, scenarios: reconcileScenarioSet(workloads, scenarios, input)};
}

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
  return (run.codec_label ?? run.codec) === state.codec && run.backend === state.backend && run.sink === state.sink;
}

export function comparable(sweep, meta) {
  const retiredKey = meta.retired || meta.key.split(".").pop();
  return !(sweep.retired || []).includes(retiredKey);
}

export function bestRun(sweep, scenario, fill, state, meta) {
  const wantHigh = meta.better === "high";
  let best = null, bestValue = null, count = 0;
  for (const run of sweep.runs) {
    if (run.scenario !== scenario || inputKey(run) !== fill) continue;
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
  const {codec, fill, backend, dtype, sink, scenarios, s3Throughput} = selection;
  return runs.filter(run => {
    if ((run.codec_label ?? run.codec) !== codec || inputKey(run) !== fill || run.dtype !== dtype || run.sink !== sink) return false;
    if (includeBackend && run.backend !== backend) return false;
    if (!scenarios.has(run.scenario)) return false;
    if (sink === "s3" && s3Throughput && String(run.s3_throughput_gbps) !== s3Throughput) return false;
    return true;
  });
}
