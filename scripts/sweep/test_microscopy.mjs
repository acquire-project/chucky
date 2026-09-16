import test from "node:test";
import assert from "node:assert/strict";
import {eligible, frontier, measurementsCsv, plottable, plotDomains, plotGroups, readState, reportRows, reportViews, resampledFrontier, writeState} from "./microscopy.mjs";

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

test("codec filters distinguish Blosc codecs from their raw controls", () => {
  const rows = [row("blosc-lz4", 2, 2), configured("lz4", 4, 1.2, {codec: "lz4", blosc_block_bytes: null}),
    configured("blosc-zstd", 1, 3, {codec: "blosc-zstd"}),
    configured("zstd", 2, 2.5, {codec: "zstd", blosc_block_bytes: null})];
  for (const codec of ["blosc-lz4", "lz4", "blosc-zstd", "zstd"]) {
    assert.deepEqual(eligible(rows, {codec}).map(value => value.id), [codec]);
    assert.equal(readState(`?codec=${codec}`, rows).codec, codec);
  }
  assert.deepEqual(eligible(rows, {codec: "lz4", block: "16384"}).map(value => value.id), ["lz4"]);
});

test("fitted axes retain sub-one folds and full observed throughput ranges", () => {
  const rows = [row("padded", 2, 0.8), row("compressed", 3, 1.2)];
  const domains = plotDomains(rows);
  assert.ok(domains.fold[0] > 0 && domains.fold[0] < 0.8);
  assert.ok(domains.fold[1] > 1.2 && domains.fold[1] < 1.5);
  assert.ok(domains.throughput[0] > 0 && domains.throughput[0] < rows[0].throughput.min);
  assert.ok(domains.throughput[1] > rows[1].throughput.max && domains.throughput[1] < 4);
});

test("narrow, single-point, and unavailable objectives have valid plot domains", () => {
  for (const rows of [[], [row("one", 2, 1)], [row("a", 2, 1), row("b", 2.01, 1.001)],
    [row("bad", NaN, 1), row("zero", 2, 0), row("negative", 2, -1)]]) {
    const domains = plotDomains(rows);
    assert.ok(domains.fold[0] > 0 && domains.fold[1] > domains.fold[0]);
    assert.ok(domains.throughput[0] >= 0 && domains.throughput[1] > domains.throughput[0]);
    for (const point of rows.filter(plottable)) {
      assert.ok(point.compression_fold > domains.fold[0] && point.compression_fold < domains.fold[1]);
      assert.ok(point.throughput.min > domains.throughput[0] && point.throughput.max < domains.throughput[1]);
    }
  }
});

test("axis matching survives URLs without changing frontier membership", () => {
  const rows = [row("a", 2, 2), row("b", 1, 1)];
  for (const axes of ["panel", "input", "all"]) {
    const state = readState(`?axes=${axes}&extent=frontier`, rows);
    assert.equal(state.axes, axes);
    assert.equal(state.extent, "frontier");
    assert.deepEqual(readState(writeState(state), rows), state);
    assert.deepEqual(frontier(rows, state).ids, new Set(["a"]));
  }
  assert.equal(readState("?axes=bogus", rows).axes, "panel");
  assert.equal(readState("?extent=bogus", rows).extent, "all");
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


test("comparison variation stays visible without suppressing its observed frontier", () => {
  const rows = [row("stable", 2, 2), row("variable", 20, 20,
    {phase: "comparison", reference: {drift: true}})];
  assert.deepEqual(frontier(rows).ids, new Set(["variable"]));
  assert.equal(resampledFrontier(rows[0]), false);
  assert.equal(resampledFrontier({...rows[0], uncertainty: {frontier_frequency: 0.2}}), true);
  assert.equal(resampledFrontier({...rows[0], uncertainty: {frontier_frequency: 0.01}}), false);
});

test("fold ranges fit inside the axes and uncertainty exports with its scope", () => {
  const value = row("varied", 2, 1.1, {compression_range: {min: 0.8, max: 1.3},
    uncertainty: {frontier_frequency: 0.4, throughput: {lower: 1.8, upper: 2.1},
      compression_fold: {lower: 0.9, upper: 1.2}, scope: "all selected settings"}});
  const domains = plotDomains([value]);
  assert.ok(domains.fold[0] < 0.8 && domains.fold[1] > 1.3);
  const csv = measurementsCsv([value], new Set());
  assert.ok(csv.includes("resampled_frontier_frequency"));
  assert.ok(csv.includes("all selected settings"));
});


test("report selection replaces superseded inputs and keeps uncovered backends", () => {
  const dataset = (id, measurements) => ({study: {id, machine: {name: "L40"}}, measurements:
    measurements.map(value => ({...value, study_id: id, condition: `${id}-${value.config.backend}`, input_label: "Image v2"}))});
  const old = dataset("old", [row("old-cpu", 2, 2), configured("old-gpu", 4, 2, {backend: "gpu"})]);
  const current = dataset("current", [configured("current-gpu", 5, 2, {backend: "gpu"})]);
  const report = [{input: "image", label: "Image", sources: [
    {study: "current", backends: ["gpu"]}, {study: "old", backends: ["cpu"]}]}];
  const selected = reportRows([old, current], report);
  assert.deepEqual(selected.map(value => value.id), ["current-gpu", "old-cpu"]);
  assert.ok(selected.every(value => value.input_label === "Image" && value.machine === "L40"));
  assert.deepEqual(selected.map(value => value.count), [3, 3]);
  assert.equal(current.measurements[0].input_label, "Image v2");
  assert.equal(frontier(selected).ids.size, 2);
  assert.equal(readState("?study=old&selected=old-gpu", selected).selected, null);
  assert.ok(!writeState(readState("?study=old", selected)).includes("study="));
  const csv = measurementsCsv(selected, new Set());
  assert.ok(csv.includes("current,current-gpu") && csv.includes("old,old-cpu"));
  assert.ok(!csv.includes("old-gpu"));
});

test("report selection fails on missing or repeated measurements", () => {
  const datasets = [{study: {id: "study", machine: {name: "L40"}}, measurements: [row("a", 2, 2)]}];
  const source = {study: "study", backends: ["cpu"]};
  const item = {input: "image", label: "Image", sources: [source]};
  assert.throws(() => reportRows(datasets, [{...item, input: "missing"}]), /Missing report measurements/);
  assert.throws(() => reportRows(datasets, [{...item, sources: [{...source, study: "missing"}]}]), /Missing report source/);
  assert.throws(() => reportRows(datasets, [{...item, sources: [source, source]}]), /Duplicate report measurements/);
});

test("machine filtering and the default full extent preserve measurements", () => {
  const rows = [row("l40", 2, 2, {machine: "L40"}), row("m4", 3, 2, {machine: "M4", condition: "other"})];
  const state = readState("?machine=M4", rows);
  assert.equal(state.extent, "all");
  assert.deepEqual(eligible(rows, state).map(value => value.id), ["m4"]);
  assert.deepEqual(readState(writeState(state), rows), state);
  assert.equal(readState("?extent=all", rows).extent, "all");
});

test("combined plots retain per-input frontiers and separate machines, backends, and sinks", () => {
  const input = (id, rate, fold, name, backend = "cpu", sink = "discard", machine = "L40") => {
    const value = configured(id, rate, fold, {input_id: name, backend, sink});
    return {...value, machine, condition: `${machine}-${name}-${backend}-${sink}`};
  };
  const rows = [input("a", 2, 2, "first"), input("dominated", 1, 1, "first"),
    input("b", 100, 100, "second"), input("gpu", 3, 2, "first", "gpu"),
    input("fs", 1, 2, "first", "cpu", "fs"), input("m4", 1, 2, "first", "cpu", "discard", "M4")];
  const groups = plotGroups(rows);
  assert.equal(groups.length, 4);
  assert.deepEqual(groups.flat().map(value => value.id).sort(), rows.map(value => value.id).sort());
  const combined = groups.find(group => group.some(value => value.id === "a"));
  assert.deepEqual(combined.map(value => value.id), ["a", "dominated", "b"]);
  assert.deepEqual(frontier(combined).ids, new Set(["a", "b"]));
  assert.deepEqual(plotGroups(eligible(rows, {input: "first", backend: "gpu"})).flat().map(value => value.id), ["gpu"]);
});

test("report views select their own observations and preserve URL state", () => {
  const datasets = ["choices", "machines"].map(id => ({study: {id, machine: {name: "host"}},
    measurements: [row(id, 2, 2, {study_id: id})]}));
  const selection = study => [{input: "image", label: "Image", sources: [{study, backends: ["cpu"]}]}];
  const views = reportViews({report: selection("choices"), views: [
    {id: "cpu-gpu", label: "CPU and GPU machines", description: "Selected settings", report: selection("machines")}]});
  assert.deepEqual(reportRows(datasets, views[0].report).map(row => row.id), ["choices"]);
  assert.deepEqual(reportRows(datasets, views[1].report).map(row => row.id), ["machines"]);
  const ids = views.map(view => view.id), rows = datasets[1].measurements;
  const state = readState("?view=cpu-gpu&input=image&sink=discard&axes=all", rows, ids);
  assert.equal(state.view, "cpu-gpu");
  assert.deepEqual(readState(writeState(state), rows, ids), state);
  assert.equal(readState("?view=unknown", rows, ids).view, "pareto");
  assert.equal(readState("", rows).view, "pareto");
  assert.equal(reportViews({}).length, 1);
});
