// Reproduce the historical 200-configuration matrix. Requires Node.js and a
// Release benchmark with --blosc-shuffle and returned shuffle/level metadata.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {spawn, execFileSync} from 'node:child_process';
import {gzipSync} from 'node:zlib';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../..');
const exe = path.resolve(process.argv[2] ?? path.join(root, 'build/bench/bench_stream_orca2_single.exe'));
const rawPath = path.join(here, 'raw-results.jsonl');
const provenancePath = path.join(here, 'provenance.json');
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const command = (name, args) => execFileSync(name, args, {cwd: root, encoding: 'utf8', windowsHide: true}).trim();
const gpu = () => command('nvidia-smi', ['--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv,noheader']);
if (fs.existsSync(rawPath) || fs.existsSync(path.join(here, 'raw-results.jsonl.gz')))
  throw new Error('Existing measurements found; use a new artifact directory.');

const configs = [];
for (const fill of ['xor', 'rand']) for (const chunk_kib of [256, 1024]) {
  for (const codec of ['lz4', 'zstd']) configs.push({fill, chunk_kib, block_kib: 0, codec, shuffle: 'none'});
  for (const codec of ['blosc-lz4', 'blosc-zstd']) for (const shuffle of ['none', 'byte', 'bit'])
    for (const block_kib of [4, 8, 16, 32, 64, 128, 256, 512, 1024])
      if (block_kib <= chunk_kib) configs.push({fill, chunk_kib, block_kib, codec, shuffle});
}
if (configs.length !== 200) throw new Error('Unexpected matrix size');
const cache = fs.readFileSync(path.join(path.dirname(exe), '../CMakeCache.txt'), 'utf8');
const build = Object.fromEntries(cache.split(/\r?\n/).filter(s => /^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_C_COMPILER|NVCOMP_LIBRARY|NVCOMP_INCLUDE_DIR):/.test(s)).map(s => [s.split(':')[0], s.slice(s.indexOf('=') + 1)]));
const nvcompHeader = fs.readFileSync(path.join(build.NVCOMP_INCLUDE_DIR,'nvcomp/version.h'),'utf8');
const nvcompVersion = ['MAJOR','MINOR','PATCH','BUILD'].map(part => {
  const match = nvcompHeader.match(new RegExp(`#define NVCOMP_VER_${part}\\s+(\\d+)`));
  if (!match) throw new Error(`Missing nvCOMP version component ${part}`);
  return match[1];
}).join('.');
const provenance = {
  start_utc: new Date().toISOString(), complete: false,
  scenario: 'orca2_single', source_commit: command('git', ['rev-parse', 'HEAD']),
  tracked_source_diff: command('git', ['diff', '--', 'bench/bench_util.c', 'bench/bench_report.c']),
  build, executable_sha256: hash(exe),
  source_sha256: Object.fromEntries(['bench/bench_util.c', 'bench/bench_report.c', 'tests/test_data.c'].map(p => [p, hash(path.join(root, p))])),
  platform: process.platform, os_version: os.version(), cpu: os.cpus()[0]?.model,
  logical_cpus: os.cpus().length, physical_ram_bytes: os.totalmem(),
  cuda_compiler: command(build.CMAKE_CUDA_COMPILER, ['--version']), nvcomp: nvcompVersion,
  gpu_before: gpu(), gpu_samples: [],
  options: {frames: 100, repeats: 3, warmups: 1, memory_budget: '6G', batch_bytes: '64M', max_threads: 3, dtype: 'u16', seed: 169},
  randomization: 'Seed 169, Mulberry32 / Fisher-Yates; fresh shuffle of canonical matrix per pass. PRNG differs from historical runner.',
  geometry: {'256': {shape: [8, 1, 128, 128], chunks_per_epoch: 576, padded_batch_bytes: 150994944}, '1024': {shape: [16, 1, 128, 256], chunks_per_epoch: 288, padded_batch_bytes: 301989888}},
  validation: 'test-test_compress_blosc_gpu passed before measurement; every run checks returned filter, level, block, shape, epoch count, batch, and bytes.',
  runs: 0, measured_runs: 0, failures: []
};
const save = () => fs.writeFileSync(provenancePath, JSON.stringify(provenance, null, 2) + '\n');
save();
let state = 169;
function random() {
  let t = state += 0x6D2B79F5;
  t = Math.imul(t ^ t >>> 15, t | 1);
  t ^= t + Math.imul(t ^ t >>> 7, t | 61);
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function shuffle(rows) {
  rows = [...rows];
  for (let i = rows.length - 1; i > 0; --i) {
    const j = Math.floor(random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  return rows;
}
function run(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(exe, args, {cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe']});
    let stdout = '', stderr = '', timed_out = false;
    child.stdout.on('data', x => stdout += x);
    child.stderr.on('data', x => stderr += x);
    const timer = setTimeout(() => { timed_out = true; child.kill(); }, 120000);
    child.on('error', e => { clearTimeout(timer); reject(e); });
    child.on('close', code => { clearTimeout(timer); resolve({code, timed_out, stdout, stderr}); });
  });
}
for (let pass = 0; pass < 4; ++pass) for (const config of shuffle(configs)) {
  const args = ['--backend', 'gpu', '--dtype', 'u16', '--fill', config.fill, '--codec', config.codec,
    '--frames', '100', '--chunk-bytes', `${config.chunk_kib}K`, '--batch-bytes', '64M',
    '--memory-budget', '6G', '--max-threads', '3', '--json'];
  if (config.block_kib) args.push('--blosc-block-bytes', `${config.block_kib}K`, '--blosc-shuffle', config.shuffle);
  const start_utc = new Date().toISOString();
  const output = await run(args);
  let result, validation_error;
  try {
    result = JSON.parse(output.stdout);
    if (output.code !== 0 || result.status !== 'pass') throw new Error('Benchmark failed');
    if (config.block_kib && (result.blosc_block_bytes !== config.block_kib * 1024 || result.blosc_shuffle !== config.shuffle || result.blosc_level !== 3)) throw new Error('Codec settings mismatch');
    const geom = provenance.geometry[config.chunk_kib];
    const dims = [...output.stderr.matchAll(/^\s+[0-3]\s+[tcyx]\s+\d+\s+(\d+)/gm)].map(m => Number(m[1]));
    if (JSON.stringify(dims) !== JSON.stringify(geom.shape)) throw new Error('Chunk shape mismatch');
    if (!output.stderr.includes(`auto-fit: ${config.chunk_kib * 1024} bytes/chunk (batch=1)`)) throw new Error('Layout or batch changed');
    if (result.chunks_per_epoch !== geom.chunks_per_epoch || result.total_chunks !== geom.chunks_per_epoch * Math.ceil(100 / geom.shape[0])) throw new Error('Epoch geometry mismatch');
    if (result.stages.memcpy.in_bytes !== 1887436800 || result.worker_threads !== 3) throw new Error('Input or thread mismatch');
    if (result.stages.compress.in_bytes !== result.total_chunks * config.chunk_kib * 1024) throw new Error('Padded input mismatch');
  } catch (e) { validation_error = String(e); }
  const record = {pass, warmup: pass === 0, config, args, start_utc, finish_utc: new Date().toISOString(), ...output, result, validation_error};
  fs.appendFileSync(rawPath, JSON.stringify(record) + '\n');
  ++provenance.runs;
  if (pass) ++provenance.measured_runs;
  if (validation_error) provenance.failures.push({pass, config, validation_error});
  if (provenance.runs % 25 === 0 || validation_error) {
    provenance.gpu_samples.push({utc: new Date().toISOString(), after_run: provenance.runs, reading: gpu()});
    save();
    console.log(`${provenance.runs}/800; measured ${provenance.measured_runs}; failures ${provenance.failures.length}; ${new Date().toISOString()}`);
  }
  if (validation_error) throw new Error(`Measurement validation failed: ${validation_error}; see raw-results.jsonl`);
}
provenance.finish_utc = new Date().toISOString();
provenance.gpu_after = gpu();
provenance.complete = provenance.runs === 800 && provenance.failures.length === 0;
const raw = fs.readFileSync(rawPath);
provenance.raw_results_sha256 = crypto.createHash('sha256').update(raw).digest('hex');
fs.writeFileSync(path.join(here, 'raw-results.jsonl.gz'), gzipSync(raw));
fs.unlinkSync(rawPath);
save();
console.log('Complete; all repetitions and logs retained in raw-results.jsonl.gz');
