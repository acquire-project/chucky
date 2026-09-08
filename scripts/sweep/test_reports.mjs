import assert from "node:assert/strict";
import {test} from "node:test";
import * as blosc from "./blosc.js";
import {bestRun, moversFor, configLabel, codecSettings, filterRuns, comparable, metricValue, inputKey, inputLabel, inputLabels, performanceSweeps, scopeWorkloads, workloadsCompatible, preferredScenario, preferredInput, reconcileInput, reconcileScenario, reconcileScenarioSet, reconcileScenarioToggle, selectScenarioGroup} from "./selection.mjs";

const run = (block, overrides = {}) => ({
  scenario: "orca2_single", codec: "blosc-zstd", fill: "xor", backend: "cpu",
  dtype: "u16", chunk_bytes_label: "256K", sink: "discard", status: "pass",
  throughput_in_gibs: 1, ...(block == null ? {} : {blosc_block_bytes: block}), ...overrides,
});

const imageWorkloadRun = input => run(null, {
  scenario: "images", fill: "images", input_id: input,
});

const WORKLOADS = {
  version: 1,
  inputs: [
    {id: "xor", scenarios: ["orca2_single", "256cube_single"], default_scenario: "orca2_single"},
    {id: "rand", scenarios: ["orca2_single", "256cube_single"], default_scenario: "orca2_single"},
    {id: "opencell-dna", scenarios: ["images"], default_scenario: "images"},
    {id: "opencell-protein", scenarios: ["images"], default_scenario: "images"},
  ],
  scenarios: [
    {id: "orca2_single", inputs: ["xor", "rand"], default_input: "xor"},
    {id: "256cube_single", inputs: ["xor", "rand"], default_input: "xor"},
    {id: "images", inputs: ["opencell-dna", "opencell-protein"], default_input: "opencell-dna"},
  ],
};

test("workload scope keeps only observed registry entries", () => {
  const scope = scopeWorkloads(WORKLOADS, [
    run(null, {scenario: "orca2_single", fill: "xor"}),
    imageWorkloadRun("opencell-protein"),
  ]);
  assert.deepEqual(scope.inputs.map(input => input.id), ["xor", "opencell-protein"]);
  assert.deepEqual(scope.scenarios.map(scenario => scenario.id), ["orca2_single", "images"]);
  assert.deepEqual(scope.scenarios[1].inputs, ["opencell-protein"]);
});

test("single scenario and input selections reconcile in both directions", () => {
  assert.equal(workloadsCompatible(WORKLOADS, "images", "opencell-protein"), true);
  assert.equal(workloadsCompatible(WORKLOADS, "orca2_single", "xor"), true);
  assert.deepEqual(
    reconcileInput(WORKLOADS, {scenario: "images", input: "opencell-dna"}, "opencell-protein"),
    {scenario: "images", input: "opencell-protein"},
  );
  assert.deepEqual(
    reconcileScenario(WORKLOADS, {scenario: "orca2_single", input: "xor"}, "256cube_single"),
    {scenario: "256cube_single", input: "xor"},
  );
  assert.deepEqual(
    reconcileInput(WORKLOADS, {scenario: "orca2_single", input: "xor"}, "opencell-protein"),
    {scenario: "images", input: "opencell-protein"},
  );
  assert.deepEqual(
    reconcileScenario(WORKLOADS, {scenario: "orca2_single", input: "xor"}, "images"),
    {scenario: "images", input: "opencell-dna"},
  );
});

test("multi-scenario reconciliation prunes incompatible checks", () => {
  assert.deepEqual(
    reconcileScenarioSet(
      WORKLOADS,
      new Set(["orca2_single", "images"]),
      "opencell-protein",
    ),
    new Set(["images"]),
  );
  assert.deepEqual(
    reconcileScenarioSet(
      WORKLOADS,
      new Set(["orca2_single", "256cube_single"]),
      "opencell-protein",
    ),
    new Set(["images"]),
  );
  const checked = reconcileScenarioToggle(
    WORKLOADS,
    {input: "xor", scenarios: new Set(["orca2_single"])},
    "images",
    true,
  );
  assert.equal(checked.input, "opencell-dna");
  assert.deepEqual(checked.scenarios, new Set(["images"]));
  assert.deepEqual(
    selectScenarioGroup(
      WORKLOADS,
      new Set(["orca2_single"]),
      ["orca2_single", "256cube_single", "images"],
      "xor",
    ),
    new Set(["orca2_single", "256cube_single"]),
  );
});

test("unavailable registered defaults fall back in registry order", () => {
  const imageScope = scopeWorkloads(WORKLOADS, [imageWorkloadRun("opencell-protein")]);
  assert.equal(preferredInput(imageScope, "images"), "opencell-protein");
  const randScope = scopeWorkloads(WORKLOADS, [
    run(null, {scenario: "256cube_single", fill: "rand"}),
  ]);
  assert.equal(preferredScenario(randScope, "rand"), "256cube_single");
});

test("performance overview excludes smoke sweeps", () => {
  const sustained = {smoke: false};
  assert.deepEqual(performanceSweeps([{smoke: true}, sustained, {}]), [sustained, {}]);
});

test("block requests distinguish unknown, null, and explicit sizes", () => {
  assert.equal(blosc.bloscBlockKey(run()), "unknown");
  assert.equal(blosc.bloscBlockKey(run(null, {blosc_block_bytes: null})), "unknown");
  assert.equal(blosc.bloscBlockKey(run(16384)), "16384");
  assert.equal(blosc.bloscBlockKey(run(null, {codec: "zstd"})), "");
  assert.equal(blosc.bloscBlockLabel("16384"), "16 KiB");
  assert.equal(blosc.bloscBlockLabel("4097"), "4097 B");
  assert.match(blosc.bloscBlockLabel("unknown"), /unknown/);
});

test("overview groups block requests while retaining configuration details", () => {
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", metric: "throughput_in_gibs"};
  const meta = {key: state.metric, better: "high"};
  const before = {runs: [run(undefined, {id: "unknown", throughput_in_gibs: 100})]};
  const after = {runs: [run(16384, {id: "16K"}), run(32768, {id: "32K", throughput_in_gibs: 200})]};
  assert.equal(bestRun(before, "orca2_single", "xor", state, meta).value, 100);
  assert.equal(bestRun(after, "orca2_single", "xor", state, meta).value, 200);
  assert.equal(moversFor({sweeps: [before, after]}, state, meta).rows.length, 0);
  before.runs.push(run(16384, {id: "16K", throughput_in_gibs: 2}));
  assert.equal(moversFor({sweeps: [before, after]}, state, meta).rows[0].pct, -50);
  assert.match(configLabel(after.runs[0]), /block 16 KiB/);
  assert.match(configLabel(before.runs[0]), /unknown/);
});

test("explorer groups block requests under the selected codec", () => {
  const selection = {codec: "blosc-zstd", fill: "xor", backend: "cpu",
    dtype: "u16", sink: "discard", scenarios: new Set(["orca2_single"])};
  const runs = [run(), run(16384), run(32768), run(16384, {backend: "gpu"})];
  assert.equal(filterRuns(runs, selection).length, 3);
  assert.equal(filterRuns(runs, selection, {includeBackend: false}).length, 4);
});

test("selection respects metric direction, retirement, and missing values", () => {
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", metric: "stages.compress_ms"};
  const meta = {key: state.metric, better: "low"};
  const sweep = {runs: [run(16384), run(16384, {stages: {compress_ms: 2}}),
    run(16384, {stages: {compress_ms: 1}}), run(16384, {stages: {compress_ms: Infinity}})]};
  assert.equal(bestRun(sweep, "orca2_single", "xor", state, meta).value, 1);
  assert.equal(bestRun(sweep, "orca2_single", "xor", state, meta).count, 2);
  assert.equal(metricValue(sweep.runs[0], state.metric), null);
  assert.equal(comparable(sweep, meta), true);
  sweep.retired = ["compress_ms"];
  assert.equal(comparable(sweep, meta), false);
  assert.deepEqual(moversFor({sweeps: [sweep]}, state, meta), {rows: [], newest: null});
});

test("codec variants share a short selector label and retain tooltip settings", () => {
  const variant = (block, shuffle, level, throughput = 1) => run(block, {
    blosc_shuffle: shuffle, blosc_level: level, codec_label: "blosc-zstd",
    id: `${block}-${shuffle}-${level}`, throughput_in_gibs: throughput,
  });
  const selected = variant(16384, "bit", 0);
  const runs = [selected, variant(32768, "bit", 0, 100),
    variant(16384, "byte", 0, 200), variant(16384, "bit", 3, 300),
    variant(4096, "byte", 0, 400), run(undefined, {codec_label: "blosc-zstd"})];
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", metric: "throughput_in_gibs"};
  const meta = {key: state.metric, better: "high"};
  const matching = filterRuns(runs, {...state, fill: "xor", dtype: "u16",
    scenarios: new Set(["orca2_single"])});
  assert.equal(matching.length, 6);
  assert.equal(bestRun({runs}, "orca2_single", "xor", state, meta).run, runs[4]);
  assert.equal(codecSettings(selected), "bit shuffle, level 0");
  assert.equal(codecSettings(run(16384)), "");
});

const imageRun = (input, overrides = {}) => run(16384, {
  scenario: "images", fill: "images", input_id: input, input_label: "Cellstate / fluorescence",
  id: "images__" + input, ...overrides,
});

test("overview separates image inputs before selecting the best run", () => {
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", metric: "throughput_in_gibs"};
  const meta = {key: state.metric, better: "high"};
  const sweep = {runs: [imageRun("pack-a", {throughput_in_gibs: 7}),
    imageRun("pack-b", {throughput_in_gibs: 1000})]};
  assert.equal(bestRun(sweep, "images", "pack-a", state, meta).value, 7);
  assert.equal(bestRun(sweep, "images", "images", state, meta), null);
  const previous = {runs: [imageRun("another-protocol", {throughput_in_gibs: 700})]};
  assert.equal(moversFor({sweeps: [previous, sweep]}, state, meta).rows.length, 0);
});

test("explorer filters image identities and retains backend overlays", () => {
  const selection = {codec: "blosc-zstd", fill: "pack-a", backend: "cpu",
    dtype: "u16", sink: "discard", scenarios: new Set(["images"])};
  const rows = [imageRun("pack-a"), imageRun("pack-b"), imageRun("pack-a", {backend: "gpu"})];
  assert.equal(filterRuns(rows, selection).length, 1);
  assert.equal(filterRuns(rows, selection, {includeBackend: false}).length, 2);
  assert.equal(inputKey(run(16384)), "xor");
  assert.equal(inputLabel(imageRun("pack-a")), "Cellstate / fluorescence");
  const labels = inputLabels(rows);
  assert.equal(labels.size, 2);
  assert.notEqual(labels.get("pack-a"), labels.get("pack-b"));
});
