import assert from "node:assert/strict";
import {test} from "node:test";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {dominates, frontier, eligible, memoryValue, defaultState, readState, writeState, measurementsCsv} from "./pareto.mjs";
import {fmtSignificant} from "./charts.js";

const row = (id, speed, fold, memory = 2, extra = {}) => ({id, experiment_id: "a", workload_id: "xor-256",
  codec: "blosc-lz4", shuffle: "bit", block_kib: 16, control: false,
  throughput_gibs: {median: speed, min: speed - .1, max: speed + .1}, compression_fold: fold,
  measured_device_gib: {median: memory, min: memory, max: memory}, estimated_device_gib: memory,
  provenance: {summary: "source.csv", summary_line: 2}, source_metrics: {}, ...extra});
const ids = (rows, options) => [...frontier(rows, options).ids].sort();

test("selected-setting values use three significant figures", () => {
  assert.equal(fmtSignificant(1.03908), "1.04");
  assert.equal(fmtSignificant(10.9564), "11.0");
  assert.equal(fmtSignificant(4.754), "4.75");
  assert.equal(fmtSignificant(0.000012345), "0.0000123");
  assert.equal(fmtSignificant(null), "—");
});

test("dominance is strict in at least one objective; exact ties remain", () => {
  assert.equal(dominates([1, 2], [1, 2]), false);
  assert.equal(dominates([2, 2], [1, 2]), true);
  assert.equal(dominates([2, 1], [1, 2]), false);
  assert.deepEqual(ids([row("a", 2, 2), row("b", 2, 2), row("c", 1, 2)]), ["a", "b"]);
  assert.deepEqual(ids([row("a", 2.0000000001, 2), row("b", 2, 2)]), ["a"]);
});

test("system, workload, and per-codec boundaries are independent", () => {
  const rs = [row("base", 1, 1), row("system", 10, 10, 1, {experiment_id: "b"}),
    row("workload", 10, 10, 1, {workload_id: "different-shape"}),
    row("codec", 2, 2, 2, {codec: "blosc-zstd"})];
  assert.equal(ids(rs).length, 4);
  assert.deepEqual(ids(rs, {mode: "cross"}), ["codec", "system", "workload"]);
});

test("eligibility and budget filters precede dominance; controls never join", () => {
  const rs = [row("small", 1, 1, 1), row("large", 2, 2, 3),
    row("raw", 100, 100, .1, {codec: "lz4", control: true, block_kib: 0})];
  assert.deepEqual(ids(rs, {budget: 2}), ["small"]);
  assert.deepEqual(ids(rs), ["large"]);
  assert.equal(eligible(rs).length, 3);
  assert.equal(eligible(rs, {systems: []}).length, 0);
  assert.deepEqual(ids(rs, {shuffles: ["none"]}), []);
  assert.deepEqual(ids(rs, {blocks: [16], codecs: ["zstd"]}), []);
  assert.deepEqual(ids([row("unknown", 10, 10, 2, {estimated_device_gib: null})], {budget: 6}), []);
});

test("memory is a view quantity, not a frontier objective", () => {
  const rs = [row("fast", 3, 3, 3, {estimated_device_gib: 4}),
    row("small", 2, 2, 1, {estimated_device_gib: 1})];
  assert.deepEqual(ids(rs, {mode: "cross"}), ["fast"]);
  assert.equal(memoryValue(rs[1]), 1);
  assert.throws(() => frontier(rs, {mode: "memory"}), /Unknown frontier mode/);
});

test("URL state round-trips empty subsets, selection, views, budget and sorting", () => {
  const state = {...defaultState(["a", "b"]), systems: [], codecs: ["zstd"], shuffles: ["bit", "byte"], blocks: [16],
    budget: 2.123456789, mode: "cross", layout: "overlay", workload: "w", view: "memory",
    selected: "a:xor & bit", extent: "fit", sort: "estimated", direction: "asc"};
  assert.deepEqual(readState(writeState(state), ["a", "b"], ["w"], [16]), state);
  const bad = readState("?mode=bad&budget=NaN&scale=linear&systems=missing&blocks=abc&selected=", ["a"], ["w"], [16]);
  assert.equal(bad.mode, "codec"); assert.equal(bad.budget, null); assert.equal("scale" in bad, false);
  assert.deepEqual(bad.systems, []); assert.deepEqual(bad.blocks, []);
});

test("CSV retains exact values, nulls, provenance, and escapes embedded fields", () => {
  const r = row("quoted,\"id", 1.1234567890123, 2, null);
  const csv = measurementsCsv([r], new Set([r.id]));
  assert.match(csv, /"quoted,""id"/); assert.match(csv, /1\.1234567890123/);
  assert.match(csv, /source\.csv/); assert.match(csv, /true,false/);
  assert.equal(measurementsCsv([], new Set()).split("\r\n").length, 2);
});

// The complete report build is the fixture, so these assertions exercise the
// same Python output the browser consumes. Run report.py before this test.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const site = path.resolve(process.env.PARETO_SITE || path.join(root, "_site"));
const manifest = JSON.parse(fs.readFileSync(path.join(root, "docs/benchmarks/datasets.json"), "utf8"));
const numericCsv = file => {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/), fields = lines.shift().split(",");
  return lines.map(line => Object.fromEntries(line.split(",").map((v,i) => [fields[i], v])));
};
const config = r => [r.fill, r.chunk_kib, r.codec, r.shuffle, r.block_kib].join("__");
for (const spec of manifest.experiments) {
  test(`${spec.label}: normalized frontiers agree with all retained numeric results`, () => {
    const data = JSON.parse(fs.readFileSync(path.join(site, "data/pareto", spec.id + ".json"), "utf8"));
    const directory = path.join(root, "docs/benchmarks", spec.directory);
    const retained = numericCsv(path.join(directory, "pareto-frontier.csv"));
    const members = mode => {
      const hit = frontier(data.measurements, {mode});
      return hit.candidates.filter(r => hit.ids.has(r.id)).map(config).sort();
    };
    assert.deepEqual(members("codec"), retained.map(config).sort());
    const yes = value => String(value).toLowerCase() === "true";
    assert.deepEqual(members("cross"), retained.filter(r => yes(r.overall_frontier ?? r.cross_codec_frontier)).map(config).sort());
    if (spec.format === "node-jsonl-v1") {
      const budgets = numericCsv(path.join(directory, "pareto-by-allocation-budget.csv"));
      for (const budget of [1.5, 2, 2.5, 3, 4, 6]) {
        const current = frontier(data.measurements, {mode: "cross", budget});
        assert.deepEqual(current.candidates.filter(r => current.ids.has(r.id)).map(config).sort(), budgets.filter(r => +r.budget_gib === budget).map(config).sort());
      }
    }
  });
}
