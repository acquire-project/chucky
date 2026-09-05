import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import {runInNewContext} from "node:vm";

const read = name => readFileSync(new URL(name, import.meta.url), "utf8");
const blosc = await import(`data:text/javascript,${encodeURIComponent(read("blosc.js"))}`);
const run = (block, overrides = {}) => ({
  scenario: "orca2_single", codec: "blosc-zstd", fill: "xor", backend: "cpu",
  dtype: "u16", chunk_bytes_label: "256K", sink: "discard", status: "pass",
  throughput_in_gibs: 1, ...(block == null ? {} : {blosc_block_bytes: block}), ...overrides,
});

function pageFunctions(page, names, globals = {}) {
  const source = read(page);
  const functions = names.map(name => {
    const match = source.match(new RegExp(`^function ${name}\\([^]*?^}`, "m"));
    assert.ok(match, `${page}: ${name}`);
    return match[0];
  }).join("\n");
  return runInNewContext(`${functions}\n({${names.join(",")}})`, {...blosc, ...globals});
}

test("report modules parse", () => {
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  for (const page of ["overview.html", "template.html"]) {
    const script = read(page).match(/<script type="module">([^]*?)<\/script>/)[1];
    assert.doesNotThrow(() => new AsyncFunction(script.replace(/^import .*;$/gm, "")));
  }
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
  const f = pageFunctions("overview.html", ["matchesSetup", "bestRun", "moversFor", "configLabel"], {
    state, comparable: () => true, metricValue: (r, key) => r[key],
    metricMeta: () => ({better: "high"}), percentChange: (a, b) => (b - a) / a * 100,
  });
  const before = {runs: [run(undefined, {id: "unknown", throughput_in_gibs: 100})]};
  const after = {runs: [run(16384, {id: "16K"}), run(32768, {id: "32K", throughput_in_gibs: 200})]};
  assert.equal(f.bestRun(before, "orca2_single", "xor"), null);
  assert.equal(f.bestRun(after, "orca2_single", "xor").value, 1);
  assert.equal(f.moversFor({sweeps: [before, after]}).rows.length, 0);
  before.runs.push(run(16384, {id: "16K", throughput_in_gibs: 2}));
  assert.equal(f.moversFor({sweeps: [before, after]}).rows[0].pct, -50);
  state.bloscBlock = "unknown";
  assert.equal(f.bestRun(before, "orca2_single", "xor").value, 100);
  assert.equal(f.bestRun(after, "orca2_single", "xor"), null);
  assert.match(f.configLabel(after.runs[0]), /block 16 KiB/);
  assert.match(f.configLabel(before.runs[0]), /unknown/);
});

test("explorer filters block requests before heatmap and line grouping", () => {
  const radios = {"codec-group": "blosc-zstd", "fill-group": "xor", "backend-group": "cpu",
    "dtype-group": "u16", "sink-group": "discard", "blosc-block-group": "16384"};
  const f = pageFunctions("template.html", ["filterRuns"], {
    runs: [run(), run(16384), run(32768), run(16384, {backend: "gpu"})],
    getRadio: key => radios[key], getCheckedScenarios: () => ["orca2_single"], allS3Throughputs: [],
  });
  assert.equal(f.filterRuns().length, 1);
  assert.equal(f.filterRuns()[0].blosc_block_bytes, 16384);
  assert.equal(f.filterRuns({includeBackend: false}).length, 2);
  radios["blosc-block-group"] = "unknown";
  assert.equal(f.filterRuns().length, 1);
  assert.equal(f.filterRuns()[0].blosc_block_bytes, undefined);
});
