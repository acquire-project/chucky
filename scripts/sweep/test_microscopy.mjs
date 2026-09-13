import test from "node:test";
import assert from "node:assert/strict";
import {eligible, frontier, measurementsCsv, readState, writeState} from "./microscopy.mjs";

function row(id, rate, fold, overrides = {}) {
  return {id, study_id: "study", condition: "cpu-discard-image", config: {
    input_id: "image", backend: "cpu", sink: "discard", codec: "blosc-lz4",
    chunk_label: "16K", blosc_block_bytes: 4096, blosc_shuffle: "bit", level: 3,
  }, throughput: {median: rate, min: rate * 0.99, max: rate * 1.01}, compression_fold: fold,
  reference: {drift: false, spread_percent: 1}, count: 3, ...overrides};
}

function configured(id, rate, fold, config) {
  const value = row(id, rate, fold);
  return {...value, config: {...value.config, ...config}};
}

test("chunk size, block size, codec, and raw controls compete together", () => {
  const rows = [row("small", 2, 2), configured("large", 3, 3, {chunk_label: "256K", blosc_block_bytes: 262144}),
    configured("raw", 4, 2, {codec: "lz4", blosc_block_bytes: null, blosc_shuffle: "none"}),
    configured("zstd", 1, 5, {codec: "blosc-zstd"}), configured("none", 6, 0.9, {codec: "none", blosc_block_bytes: null})];
  assert.deepEqual(frontier(rows).ids, new Set(["large", "raw", "zstd", "none"]));
});

test("raw controls remain visible when Blosc blocks are filtered", () => {
  const rows = [row("block4", 2, 2), configured("block16", 3, 3, {blosc_block_bytes: 16384}),
    configured("raw", 4, 1.2, {codec: "lz4", blosc_block_bytes: null, blosc_shuffle: "none"})];
  assert.deepEqual(eligible(rows, {block: "16384"}).map(value => value.id), ["block16", "raw"]);
  assert.deepEqual(frontier(rows, {block: "4096"}).ids, new Set(["block4", "raw"]));
});

test("different study/input/backend/sink conditions never dominate one another", () => {
  const rows = [row("a", 2, 2), row("b", 100, 100, {condition: "gpu-discard-image"}),
    row("c", 200, 200, {condition: "gpu-fs-image"}), row("d", 300, 300, {condition: "gpu-fs-other"}),
    row("e", 400, 400, {condition: "other-study-gpu-fs-other"})];
  assert.equal(frontier(rows).ids.size, 5);
});

test("drifting groups remain visible but cannot claim a frontier", () => {
  const rows = [row("stable", 2, 2), row("drifting", 20, 20, {reference: {drift: true}})];
  const result = frontier(rows);
  assert.equal(result.candidates.length, 2);
  assert.deepEqual(result.ids, new Set(["stable"]));
});

test("ties remain candidates and nonfinite objectives are excluded", () => {
  assert.deepEqual(frontier([row("a", 2, 2), row("b", 2, 2), row("bad", NaN, 3)]).ids, new Set(["a", "b"]));
});

test("filter and selection URLs round trip and reject unknown options", () => {
  const rows = [row("a", 2, 2), configured("b", 3, 3, {chunk_label: "256K"})];
  const state = readState("?selected=a&chunk=256K&block=4096&input=image", rows);
  assert.deepEqual(readState(writeState(state), rows), state);
  assert.equal(readState("?selected=missing&chunk=bogus", rows).selected, null);
  assert.equal(readState("?selected=missing&chunk=bogus", rows).chunk, "all");
});

test("CSV preserves exact logical values and escapes spreadsheet formulas", () => {
  const value = row("=SUM(1)", 1.234567891, 2.3456789);
  const csv = measurementsCsv([value], new Set([value.id]));
  assert.ok(csv.includes("1.234567891"));
  assert.ok(csv.includes("'=SUM(1)"));
  assert.ok(csv.includes("logical_gibs_median"));
  assert.ok(csv.includes("logical_compression_fold"));
});
