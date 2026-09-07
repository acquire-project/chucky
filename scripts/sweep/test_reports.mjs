import assert from "node:assert/strict";
import {test} from "node:test";
import * as blosc from "./blosc.js";
import {bestRun, moversFor, configLabel, filterRuns, comparable, metricValue} from "./selection.mjs";

const run = (block, overrides = {}) => ({
  scenario: "orca2_single", codec: "blosc-zstd", fill: "xor", backend: "cpu",
  dtype: "u16", chunk_bytes_label: "256K", sink: "discard", status: "pass",
  throughput_in_gibs: 1, ...(block == null ? {} : {blosc_block_bytes: block}), ...overrides,
});

test("block requests distinguish unknown, null, and explicit sizes", () => {
  assert.equal(blosc.bloscBlockKey(run()), "unknown");
  assert.equal(blosc.bloscBlockKey(run(null, {blosc_block_bytes: null})), "unknown");
  assert.equal(blosc.bloscBlockKey(run(16384)), "16384");
  assert.equal(blosc.bloscBlockKey(run(null, {codec: "zstd"})), "");
  assert.deepEqual(blosc.bloscBlockChoices([run(), run(32768), run(16384), run(16384)], "blosc-zstd"),
    ["16384", "32768", "unknown"]);
  assert.deepEqual(blosc.bloscBlockChoices([run()], "zstd"), []);
  assert.equal(blosc.bloscBlockLabel("16384"), "16 KiB");
  assert.equal(blosc.bloscBlockLabel("4097"), "4097 B");
  assert.match(blosc.bloscBlockLabel("unknown"), /unknown/);
});

test("overview trends and movers require matching block requests", () => {
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", metric: "throughput_in_gibs",
    bloscBlock: "16384"};
  const meta = {key: state.metric, better: "high"};
  const before = {runs: [run(undefined, {id: "unknown", throughput_in_gibs: 100})]};
  const after = {runs: [run(16384, {id: "16K"}), run(32768, {id: "32K", throughput_in_gibs: 200})]};
  assert.equal(bestRun(before, "orca2_single", "xor", state, meta), null);
  assert.equal(bestRun(after, "orca2_single", "xor", state, meta).value, 1);
  assert.equal(moversFor({sweeps: [before, after]}, state, meta).rows.length, 0);
  before.runs.push(run(16384, {id: "16K", throughput_in_gibs: 2}));
  assert.equal(moversFor({sweeps: [before, after]}, state, meta).rows[0].pct, -50);
  state.bloscBlock = "unknown";
  assert.equal(bestRun(before, "orca2_single", "xor", state, meta).value, 100);
  assert.equal(bestRun(after, "orca2_single", "xor", state, meta), null);
  assert.match(configLabel(after.runs[0]), /block 16 KiB/);
  assert.match(configLabel(before.runs[0]), /unknown/);
});

test("explorer filters block requests before heatmap and line grouping", () => {
  const selection = {codec: "blosc-zstd", fill: "xor", backend: "cpu",
    dtype: "u16", sink: "discard", bloscBlock: "16384", scenarios: new Set(["orca2_single"])};
  const runs = [run(), run(16384), run(32768), run(16384, {backend: "gpu"})];
  assert.equal(filterRuns(runs, selection).length, 1);
  assert.equal(filterRuns(runs, selection)[0].blosc_block_bytes, 16384);
  assert.equal(filterRuns(runs, selection, {includeBackend: false}).length, 2);
  selection.bloscBlock = "unknown";
  assert.equal(filterRuns(runs, selection).length, 1);
  assert.equal(filterRuns(runs, selection)[0].blosc_block_bytes, undefined);
});

test("selection respects metric direction, retirement, and missing values", () => {
  const state = {codec: "blosc-zstd", backend: "cpu", sink: "discard", bloscBlock: "16384",
    metric: "stages.compress_ms"};
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

test("codec variants and block requests stay distinct throughout report selection", () => {
  const variant = (block, shuffle, level, throughput = 1) => run(block, {
    blosc_shuffle: shuffle, blosc_level: level, codec_label: `blosc-zstd (${shuffle}, level ${level})`,
    id: `${block}-${shuffle}-${level}`, throughput_in_gibs: throughput,
  });
  const selected = variant(16384, "bit", 0);
  const runs = [selected, variant(32768, "bit", 0, 100),
    variant(16384, "byte", 0, 200), variant(16384, "bit", 3, 300),
    variant(4096, "byte", 0, 400), run(undefined, {codec_label: "blosc-zstd"})];
  const state = {codec: selected.codec_label, backend: "cpu", sink: "discard",
    bloscBlock: "16384", metric: "throughput_in_gibs"};
  const meta = {key: state.metric, better: "high"};
  assert.deepEqual(blosc.bloscBlockChoices(runs, state.codec), ["16384", "32768"]);
  assert.deepEqual(filterRuns(runs, {...state, fill: "xor", dtype: "u16",
    scenarios: new Set(["orca2_single"])}), [selected]);
  assert.equal(bestRun({runs}, "orca2_single", "xor", state, meta).run, selected);
  const before = {runs: runs.slice(1)};
  const after = {runs};
  assert.deepEqual(moversFor({sweeps: [before, after]}, state, meta).rows, []);
  before.runs.push({...selected, throughput_in_gibs: 2});
  const rows = moversFor({sweeps: [before, after]}, state, meta).rows;
  assert.equal(rows.length, 1);
  assert.equal(rows[0].pct, -50);
  assert.equal(rows[0].run, selected);
});
