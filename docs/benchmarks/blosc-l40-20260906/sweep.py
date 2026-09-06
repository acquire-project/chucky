import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'repo'
HEAD = '2c131223a6d1a915df45b0d393bc20865bd74f92'
REPEATS = 5
HISTORY = REPO / 'docs/benchmarks/blosc-rtx5070-20260905'
BENCH = REPO / 'build/bench/bench_stream_orca2_single'
GIB = 1024 ** 3
BINARIES = [
    'bench/bench_stream_orca2_single', 'bench/bench_stream_smallepoch_single',
    'tests/test_compress_cpu', 'tests/test_compress_blosc_gpu',
    'tests/test_multiarray_gpu', 'tests/test_zarr_readback_gpu',
    'tests/test_bench_memory',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def check_source():
    if command('git', '-C', str(REPO), 'rev-parse', 'HEAD') != HEAD:
        raise RuntimeError('PR source revision changed')
    diff = subprocess.check_output(['git', '-C', str(REPO), 'diff', '--binary'])
    if diff != (ROOT / 'benchmark-controls.patch').read_bytes():
        raise RuntimeError('Evaluation source patch changed')
    return hashlib.sha256(diff).hexdigest()


def matrix():
    rows = []
    for fill in ('rand', 'xor'):
        for chunk in (256, 1024):
            blocks = [block for block in (4, 8, 16, 32, 64, 128, 256, 512, 1024) if block <= chunk]
            for codec in ('blosc-lz4', 'blosc-zstd', 'lz4', 'zstd'):
                blosc = codec.startswith('blosc-')
                for block in blocks if blosc else (0,):
                    for shuffle in ('none', 'byte', 'bit') if blosc else ('none',):
                        row = dict(fill=fill, chunk_kib=chunk, block_kib=block,
                                   codec=codec, shuffle=shuffle,
                                   level=3 if blosc else (1 if codec == 'lz4' else 0))
                        row['id'] = '__'.join(str(row[k]) for k in (
                            'fill', 'chunk_kib', 'block_kib', 'codec', 'shuffle'))
                        rows.append(row)
    return rows


def bench_command(row):
    args = [str(BENCH), '--backend', 'gpu', '--dtype', 'u16', '--frames', '100',
            '--memory-budget', '6G', '--batch-bytes', '64M', '--max-threads', '3',
            '--append-elements', str(32 * 1024 * 1024), '--json',
            '--fill', row['fill'], '--chunk-bytes', str(row['chunk_kib'] * 1024),
            '--codec', row['codec'], '--shuffle', row['shuffle'],
            '--level', str(row['level'])]
    if row['block_kib']:
        args.extend(['--blosc-block-bytes', str(row['block_kib'] * 1024)])
    return args


def prepare():
    rows = matrix()
    baseline = list(csv.DictReader((HISTORY / 'summary.csv').open()))
    keys = ('fill', 'chunk_kib', 'block_kib', 'codec', 'shuffle')
    historical = {tuple(str(r[k]) for k in keys): r for r in baseline}
    for row in rows:
        if tuple(str(row[k]) for k in keys) not in historical:
            raise RuntimeError(f'No historical match: {row}')
    manifest = dict(source_commit=HEAD, patch_sha256=check_source(),
                    scenario='orca2_single', frames=100, warmups=1, repeats=REPEATS,
                    harness_sha256=sha(Path(__file__)),
                    shuffle_seed=169, memory_budget_bytes=6 * GIB,
                    target_batch_bytes=64 * 1024 ** 2, max_threads=3,
                    append_elements=32 * 1024 ** 2, dtype='u16', sink='discard',
                    configurations=[dict(r, command=bench_command(r)) for r in rows])
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(rows)} configurations; {len(rows) * (REPEATS + 1)} executions; all historical matches present')
    print('192 Blosc configurations plus 8 raw controls; five measured repetitions each')


def record_build():
    info = dict(source_commit=HEAD, patch_sha256=check_source(), built_utc=now(),
                node=command('hostname'), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                binaries={name: sha(REPO / 'build' / name) for name in BINARIES},
                manifest_sha256=sha(ROOT / 'manifest.json'),
                harness_sha256=sha(Path(__file__)),
                nvcomp_version='5.3.0.16',
                nvcomp_library='/mnt/main0/home/nclack/opt/nvcomp-5.3.0.16/lib/libnvcomp_static.a',
                nvcc=command('/mnt/main0/home/nclack/.pixi/envs/cuda-toolkit/bin/nvcc', '--version'))
    info['cmake_cache'] = [line for line in (REPO / 'build/CMakeCache.txt').read_text().splitlines()
                           if re.match(r'(CMAKE_(BUILD_TYPE|C_COMPILER|CXX_COMPILER|CUDA.*)|NVCOMP_.*):', line)]
    (ROOT / 'build.json').write_text(json.dumps(info, indent=2) + '\n')
    print('Recorded successful build and binary hashes')


def check_build():
    info = json.loads((ROOT / 'build.json').read_text())
    if info['patch_sha256'] != check_source() or info['manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('Build inputs changed')
    if info['harness_sha256'] != sha(Path(__file__)):
        raise RuntimeError('Benchmark runner changed after the build')
    for name, digest in info['binaries'].items():
        if sha(REPO / 'build' / name) != digest:
            raise RuntimeError(f'Binary changed: {name}')


def verify(row, data, stderr):
    if data.get('status') != 'pass':
        raise RuntimeError('Benchmark did not pass')
    for key in ('shuffle', 'level'):
        if data.get(key) != row[key]:
            raise RuntimeError(f'Returned {key} does not match request')
    if row['block_kib'] and data.get('blosc_block_bytes') != row['block_kib'] * 1024:
        raise RuntimeError('Returned Blosc block request differs')
    dims = re.findall(r'^\s*[0-3]\s+([tcyx])\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+[Y.]\s*$', stderr, re.M)
    expected_shape = [8, 1, 128, 128] if row['chunk_kib'] == 256 else [16, 1, 128, 256]
    if [int(d[2]) for d in dims] != expected_shape or [d[0] for d in dims] != list('tcyx'):
        raise RuntimeError(f'Actual chunk shape differs: {dims}')
    if [int(d[1]) for d in dims] != [100, 2, 2048, 2304]:
        raise RuntimeError('Input shape differs')
    geometry = re.search(r'chunks:\s+(\d+)/epoch,\s+(\d+) total \((\d+) LOD levels, batch=(\d+)\)', stderr)
    expected_chunks = 576 if row['chunk_kib'] == 256 else 288
    if not geometry or tuple(map(int, geometry.groups())) != (expected_chunks, expected_chunks, 1, 1):
        raise RuntimeError(f'Actual batch geometry differs: {geometry.group(0) if geometry else None}')
    expected_total = math.ceil(100 / expected_shape[0]) * expected_chunks
    if data['chunks_per_epoch'] != expected_chunks or data['total_chunks'] != expected_total:
        raise RuntimeError('Measured chunk count differs')
    if not math.isclose(data['input_gib'], 100 * 2 * 2048 * 2304 * 2 / GIB, rel_tol=1e-5):
        raise RuntimeError('Ingested byte count differs')
    if data['worker_threads'] != 3:
        raise RuntimeError('Worker count differs')
    if not 0 < data['memory_estimate_total_bytes'] <= 6 * GIB:
        raise RuntimeError('Device allocation estimate exceeds the fixed budget')
    if not data['memory_device_used_bytes'] or data['memory_device_overhead_bytes'] is None:
        raise RuntimeError('Observed device memory is unavailable')
    return dict(chunk_shape=expected_shape, chunks_per_epoch=expected_chunks,
                epochs_per_batch=1, padded_batch_bytes=expected_chunks * row['chunk_kib'] * 1024)


def gpu_info():
    return command('nvidia-smi', '--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total,memory.used,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem', '--format=csv')


def run():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU measurements require a Slurm allocation')
    check_build()
    names = command('nvidia-smi', '--query-gpu=name', '--format=csv,noheader').splitlines()
    if len(names) != 1 or names[0].strip() != 'NVIDIA L40':
        raise RuntimeError(f'Expected one L40: {names}')
    raw = ROOT / 'raw'
    raw.mkdir(exist_ok=False)
    provenance = dict(start_utc=now(), node=command('hostname'),
                      slurm_job_id=os.environ['SLURM_JOB_ID'],
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu_before=gpu_info(), build=json.loads((ROOT / 'build.json').read_text()),
                      manifest=json.loads((ROOT / 'manifest.json').read_text()), complete=False)
    (ROOT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    rng = random.Random(169)
    rows = matrix()
    completed = 0
    try:
        with (ROOT / 'results.jsonl').open('x') as output:
            for repeat in range(REPEATS + 1):
                order = list(rows)
                rng.shuffle(order)
                for row in order:
                    label = f'{repeat}__{row["id"]}'
                    started = time.monotonic()
                    result = subprocess.run(bench_command(row), capture_output=True, text=True, timeout=120)
                    (raw / f'{label}.json').write_text(result.stdout)
                    (raw / f'{label}.log').write_text(result.stderr)
                    if result.returncode:
                        raise RuntimeError(f'{label}: exit {result.returncode}; see raw logs')
                    data = json.loads(result.stdout)
                    actual = verify(row, data, result.stderr)
                    record = dict(config=row, repeat=repeat, warmup=repeat == 0, utc=now(),
                                  elapsed_s=time.monotonic() - started, actual=actual, result=data)
                    output.write(json.dumps(record) + '\n')
                    output.flush()
                    completed += 1
                    print(f'{completed}/{len(rows) * (REPEATS + 1)} {label}: '
                          f'{data["throughput_in_gibs"]:.3f} GiB/s, fold {data["compression_fold"]:.5g}', flush=True)
        provenance['complete'] = True
    finally:
        provenance.update(finish_utc=now(), completed=completed, gpu_after=gpu_info())
        (ROOT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    summarize()


def summarize():
    records = [json.loads(line) for line in (ROOT / 'results.jsonl').read_text().splitlines()]
    baseline = list(csv.DictReader((HISTORY / 'summary.csv').open()))
    keys = ('fill', 'chunk_kib', 'block_kib', 'codec', 'shuffle')
    old = {tuple(str(r[k]) for k in keys): r for r in baseline}
    summaries = []
    for row in matrix():
        results = [r['result'] for r in records if not r['warmup'] and r['config'] == row]
        if len(results) != REPEATS:
            raise RuntimeError(f'Expected {REPEATS} measurements for {row["id"]}, got {len(results)}')
        speeds = [r['throughput_in_gibs'] for r in results]
        folds = [r['compression_fold'] for r in results]
        if len(set(folds)) != 1:
            raise RuntimeError(f'Compression fold changed between repetitions: {row["id"]}')
        if len({r['memory_estimate_total_bytes'] for r in results}) != 1:
            raise RuntimeError(f'Allocation estimate changed across repeats: {row["id"]}')
        previous = old[tuple(str(row[k]) for k in keys)]
        speed = statistics.median(speeds)
        summaries.append(dict(row, repeats=REPEATS, throughput_median_gibs=speed,
                              throughput_min_gibs=min(speeds), throughput_max_gibs=max(speeds),
                              throughput_span_pct=100 * (max(speeds) - min(speeds)) / speed,
                              compression_fold=folds[0],
                              device_gib=statistics.median(r['memory_device_used_bytes'] for r in results) / GIB,
                              estimated_device_gib=statistics.median(r['memory_estimate_total_bytes'] for r in results) / GIB,
                              estimated_device_bytes=results[0]['memory_estimate_total_bytes'],
                              device_overhead_bytes=statistics.median(r['memory_device_overhead_bytes'] for r in results),
                              compress_total_ms=statistics.median(r['stages']['compress']['total_ms'] for r in results),
                              laptop_throughput_gibs=float(previous['throughput_median_gibs']),
                              laptop_throughput_min_gibs=float(previous['throughput_min_gibs']),
                              laptop_throughput_max_gibs=float(previous['throughput_max_gibs']),
                              laptop_fold=float(previous['compression_fold']),
                              laptop_device_gib=float(previous['device_gib']),
                              speed_relative_to_laptop=speed / float(previous['throughput_median_gibs']),
                              fold_change_pct=100 * (folds[0] / float(previous['compression_fold']) - 1)))
    for row in summaries:
        candidates = [r for r in summaries if (r['fill'], r['chunk_kib'], r['codec']) ==
                      (row['fill'], row['chunk_kib'], row['codec'])]
        def dominated(other, speed_key, fold_key):
            return (other[speed_key] >= row[speed_key] and other[fold_key] >= row[fold_key]
                    and (other[speed_key] > row[speed_key] or other[fold_key] > row[fold_key]))
        row['l40_frontier_in_subset'] = not any(dominated(r, 'throughput_median_gibs', 'compression_fold') for r in candidates)
        row['laptop_frontier_in_subset'] = not any(dominated(r, 'laptop_throughput_gibs', 'laptop_fold') for r in candidates)
    with (ROOT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fields = ['fill', 'chunk_kib', 'block_kib', 'codec', 'shuffle', 'level', 'repeats',
              'throughput_median_gibs', 'throughput_min_gibs', 'throughput_max_gibs',
              'throughput_span_pct', 'compression_fold', 'compress_total_ms',
              'device_gib', 'estimated_device_gib', 'estimated_device_bytes', 'device_overhead_bytes']
    with (ROOT / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    samples = []
    for record in records:
        result = record['result']
        samples.append(dict(record['config'], repeat=record['repeat'], warmup=record['warmup'],
                            utc=record['utc'], elapsed_s=record['elapsed_s'],
                            **{key: result[key] for key in (
                                'throughput_in_gibs', 'compression_fold', 'input_gib', 'compressed_gib',
                                'wall_s', 'init_s', 'flush_s', 'chunks_per_epoch', 'total_chunks',
                                'worker_threads', 'memory_device_used_bytes', 'memory_estimate_total_bytes',
                                'memory_device_overhead_bytes')},
                            compress_total_ms=result['stages']['compress']['total_ms']))
    with (ROOT / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(samples)
    print(f'Wrote {len(summaries)} configuration summaries and {len(samples)} individual executions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check-source', 'record-build', 'check-build', 'run', 'summarize'])
    action = parser.parse_args().action
    {'prepare': prepare, 'check-source': check_source, 'record-build': record_build,
     'check-build': check_build, 'run': run, 'summarize': summarize}[action]()
