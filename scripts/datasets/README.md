# Microscopy image replay

`bench_stream_images` preloads a small pack of uint16 images into host RAM,
then repeats its planes in manifest order. Loading and GPU context setup finish
before the throughput timer starts. The measured interval includes the final
pipeline drain and sink flush. The buffer stays alive until the stream is destroyed.

Image packs live separately, conventionally at `~/data/chucky-benchmarks/`.
The corpus is limited to 256 MiB decoded, including heldout fields. Git contains
provenance and manifests; git-annex holds binary packs with SHA256 keys.
No qualified raw `v1` exists yet.

The Cellstate proof of concept uses a separate frozen corpus. Its lock is
included here; replay requires a saved copy of `cellstate-poc-v1`. Its images
are not part of the public corpus repository.

## Cellstate proof of concept

The provisional `cellstate-poc-v1` release has four paired fields from three wells:
four mCherry and four brightfield 2048×2048 uint16 planes at middle Z=3. Each modality
pack is 32 MiB, for 64 MiB total. It preserves stored pixels, while raw acquisition
provenance remains unresolved. The explicit plan keeps all eight images without a
holdout or a claim of representative coverage.

In a saved corpus clone, check out the tag and retrieve
`data/cellstate-poc-v1/` with git-annex, then:

```sh
python scripts/datasets/run.py verify --corpus /path/to/cellstate-poc-v1 \
  --lock bench/datasets/cellstate-poc.lock.json --allow-provisional
python scripts/datasets/run.py run --corpus /path/to/cellstate-poc-v1 \
  --lock bench/datasets/cellstate-poc.lock.json --allow-provisional \
  --executable build-gpu/bench/bench_stream_images --machine reef-l40 \
  --backends cpu gpu --output bench/results/images/reef-l40-cellstate-poc
```

`--allow-provisional` permits only an explicitly provisional manifest with hashed
provenance documenting the missing evidence. Pack and plane checksums remain
mandatory, and this option cannot qualify a source in a raw release. Use this flag
and the same lock with `export.py` for materialized copies on auk or oreb.

The summary reports the minimum, median, maximum, and range divided by median for
throughput across measured repetitions. A range within 5% is the repeatability
target for this experiment. Representativeness stays inconclusive for provisional
runs regardless of repeatability.

To reproduce extraction on a compute node:

```sh
python scripts/datasets/extract.py survey \
  --plan /path/to/cellstate-poc-v1/cellstate-poc-plan.json \
  --candidates-per-modality 4 --allow-provisional --output ~/tmp/cellstate-poc-survey
python scripts/datasets/extract.py build --survey ~/tmp/cellstate-poc-survey \
  --corpus /path/to/empty-corpus-with-provenance --allow-provisional
```

A provisional survey places all planned fields in core and retains every candidate.
Building refuses an existing manifest; frozen releases are not overwritten.

## Build and verify

Use the repository's normal CPU or CUDA build configuration. The new target is
`bench_stream_images`; a CPU-only build is sufficient for CPU comparisons.

```sh
cmake --build build-cpu --target bench_stream_images test_bench_input
```

On Reef, builds, tests, image extraction, checksum scans, and benchmarks run on
approved Slurm compute allocations. The prepared build job is
`~/tmp/2026-09-06-chucky-bench/cpu-build.sh`.

The verifier checks the pinned manifest, evidence files, exact pack lengths,
pack SHA256, every plane SHA256, dtype, shape, size cap, and field separation.
Missing annex content produces a `git annex get data/` instruction. Git and annex
are unnecessary at runtime when materialized files with the same hashes are available.

## Run a pilot

After the corpus has been qualified and pinned:

```sh
python scripts/datasets/run.py run \
  --corpus ~/data/chucky-benchmarks \
  --executable build-gpu/bench/bench_stream_images \
  --machine reef-l40 --backends cpu gpu --split all \
  --output bench/results/images/reef-l40-v1
```

Defaults are one warmup and three measured process executions per pack, backend,
and codec. Each execution streams at least 8 GiB, rounded up to whole frames.
The profiles are none, Zstd level 3, Blosc-LZ4 level 3, and Blosc-Zstd level 3.
Both Blosc profiles use bitshuffle and 16 KiB blocks. The GPU path uses nvCOMP's
default compression modes; its nonzero requested levels are recorded as hints,
including Zstd. CPU levels and GPU modes are not equivalent algorithm settings.

The fixed geometry is T/Y/X chunks `1/256/256`, a 64 MiB batch target, four
workers, one scale, and the discard sink. The runner checks the reported actual
batch and shard geometry across codecs and backends. It fails on a mismatch.
Images retain native dimensions. The replay buffer is zero-padded to whole
chunks before timing; source bytes and padded bytes are reported separately.
Throughput uses native image bytes, while compressed traffic includes padding.
A memory constraint causes failure instead of an automatic geometry change.

Results are `results.json`, `summary.csv`, and per-execution logs.
They live below `bench/results/images/`. The existing sweep report discovers
these image results alongside ordinary sweeps:

```sh
uv run scripts/sweep/report.py --results-dir bench/results/ -o _site --serve
```

The [reporting workflow](../sweep/README.md#microscopy-image-inputs) shows medians,
repeat spread, and stage details, with separate inputs for each corpus and pack.
JSON includes input and output bytes, actual layouts,
initialization/loading/drain times, memory measurements, image order, corpus
identity, executable hash, source hashes, and machine information.
Compiler, CUDA, and codec-header versions are collected from the executable's
CMake build when available. Use `--toolchain toolchain.json` to attach a saved
build record; `environment.py` creates one and records the source content hash.

`logical_compression_fold` is logical image bytes divided by bytes sent to
the sink. The denominator includes sink traffic and its alignment/index overhead.
The existing `compression_fold` also counts padded input chunks; prefer the
logical ratio for these images. The discard sink's alignment is 4096 bytes.

For a short functional check, add
`--smoke --min-gib 0.016 --repeats 1 --profiles none`.
Smoke runs are always marked inconclusive for representativeness.

The core/heldout assessment compares matched modality, source group, shape,
backend, codec, and actual geometry. The target is at most 10% difference in
logical compression ratio and median throughput. Larger differences request
a revised selection. Missing pairs, incomplete repetitions, and smoke runs
remain inconclusive. Small heldout sets do not establish broad modality coverage.

## Reuse on auk and oreb

Clone the corpus over SSH from Reef and retrieve `data/v1/` as described in the
data repository README. Run the same chucky source revision and the same corpus
pin on both machines. The runner uses Python 3.10 or newer and no image libraries.

For native Windows with git-annex, use an unlocked adjusted branch:

```powershell
git checkout -b benchmark-v1 v1
git annex adjust --unlock
git annex get data/v1/
python C:/src/chucky/scripts/datasets/run.py verify --corpus .
```

The adjusted Git HEAD is recorded separately from the pinned release; matching
manifest and pixel checksums establish the input identity.
[Git-annex documents this branch mode](https://git-annex.branchable.com/git-annex-adjust/).

To transfer normal files instead, export on a compute node after retrieving
annex content:

```sh
python scripts/datasets/export.py --corpus ~/data/chucky-benchmarks \
  --output ~/tmp/chucky-benchmarks-v1.zip
```

Unpack the ZIP on either machine and pass that directory as `--corpus`.
On oreb, point `--executable` at the native Windows `bench_stream_images.exe`.
Set `--machine auk` or `--machine oreb` and write each run to a new output directory.

Compare complete runs:

```sh
python scripts/datasets/run.py compare \
  bench/results/images/auk-v1/results.json \
  bench/results/images/oreb-v1/results.json \
  --output bench/results/images/auk-oreb-v1.json
```

The comparison refuses different input hashes, protocols, source revisions,
benchmark source hashes, or actual layouts. Toolchain versions are retained:
different CUDA/nvCOMP versions can affect both compression ratio and throughput,
so such a comparison includes software differences as well as hardware.

## Extract and check the corpus

Install the optional packages in a separate environment:

```sh
uv venv --python 3.12 ~/tmp/chucky-images-env
uv pip install --python ~/tmp/chucky-images-env/bin/python \
  -r scripts/datasets/requirements-extract.txt
```

`extract.py survey` reads the explicit candidate plan and collects brightness,
entropy, adjacent-pixel differences, and compression statistics using the
benchmark chunk/filter settings. It reads full native planes, checks every
required Zarr v2 chunk, and refuses float conversions or unknown filters.
TIFF inputs must be monochrome uint16 pages with an approved lossless codec.

The survey assigns complete fields to core or heldout sets before selection.
Selection chooses actual images that best cover the measured feature values
within each modality, split, source, and shape. This is deterministic medoid
selection: each chosen image represents nearby candidates in feature space.
Default counts are 12 core and 4 heldout images per modality; the native byte
budget can reduce counts while retaining each source/shape group.
Both original coordinates and per-plane hashes are retained.

`extract.py build` requires a completed survey and, for a raw release, refuses
any source whose raw provenance is unverified. It saves the survey under `provenance/`; the verifier
and plain-file export retain and check that selection evidence.
A source's lossless container codec alone does not prove unchanged camera values.
Record an original acquisition, reviewed conversion code, or pixel comparison.

```sh
python scripts/datasets/extract.py survey \
  --plan ~/data/chucky-benchmarks/source-plan.json \
  --output ~/tmp/chucky-image-survey
python scripts/datasets/extract.py build --survey ~/tmp/chucky-image-survey \
  --corpus ~/data/chucky-benchmarks
```

The standalone C interface takes a materialized pack directly:

```sh
build-cpu/bench/bench_stream_images --input images.raw \
  --width 2048 --height 2048 --dtype u16 --frames 1024 \
  --codec blosc-zstd --codec-level 3 --shuffle bit --blosc-block-bytes 16K \
  --batch-bytes 64M --max-threads 4 --json
```

Use the Python runner for reported results; the C binary checks sizes but has
no external manifest against which to check a whole-frame truncation.

`test_bench_input` verifies the exact repeated sequence across append sizes,
wraps, partial consumption, and a partial final cycle.
`test_manifest.py` covers corruption, missing content, invalid manifests,
field leakage, and file-copy equivalence. `test_extract.py` checks exact plane
selection, missing chunks, malformed chunks, float rejection, and full synthetic
extraction. `test_storage.py` checks real annex clone/get/fsck and materialized
export. `test_runner.py` checks JSON/CSV output and comparison identity; set
`CHUCKY_IMAGE_BENCH` to the built executable to run it.
`verify_output.py` independently opens saved outputs with Zarr Python and
compares every pixel for all four profiles and two append sizes on each backend.
With `--corpus`, `--lock`, and the applicable provenance flag, it also replays each
real pack for two complete cycles plus one plane (nine frames for this corpus),
compares every saved pixel, and checks the stored Blosc block and shuffle settings.

The independent reader follows the
[Zarr Python API](https://zarr.readthedocs.io/en/stable/api/zarr/) and
[Tifffile's page-reading interface](https://github.com/cgohlke/tifffile).

## Corpus locks and local paths

Lock files stay in `bench/datasets/`. The lock pins a corpus revision and the
SHA256 of `manifest.json`. The manifest records relative pack paths, independent
SHA256 hashes of complete packs and individual planes, and hashes of provenance
and survey files. Verification reads and hashes the content directly.

Git-annex currently uses SHA256 keys, so a pack's manifest digest equals the
digest in its annex key. Verification does not parse annex keys or depend on how
annex names or locates its objects. It works with locked annex links, unlocked
files, and materialized exports.

The verifier resolves each pack path locally and retains the resolved path for
replay and export. `resolved_pack_paths` in the corpus record and `input_path` in each
execution show what was opened. Machine-specific resolved paths do not enter the
lock or comparison identity.
