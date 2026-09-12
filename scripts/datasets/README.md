# Microscopy image replay

`bench_stream_images` preloads a small pack of uint8, uint16, or float32 images
into host RAM,
then repeats its planes in manifest order through the same driver as generated
inputs. Loading, padding, and context setup precede warmup. Warmup drains and
resets metrics on that pipeline; measurement includes final drain and close. The buffer stays alive until the stream is destroyed.

The default metadata checkout is the `bench/data/microscopy` Git submodule;
git-annex holds its binary assets with SHA256 keys. [`bench/data.json`](../../bench/data.json)
declares the accepted manifest version, selects the assets used by Chucky, and
maps them to report inputs. Extra assets and unavailable unselected annex content
do not affect a sweep.

The Cellstate proof of concept uses a separate saved corpus and is not registered
as a default data source.


## Registered microscopy examples

The default `microscopy-core` selection includes all five datasets in the
format-2 collection. OpenCell contributes two packs, for six inputs in total:

| Dataset | Input IDs | Pixel type |
| --- | --- | --- |
| OpenCell | `opencell-dna`, `opencell-protein` | uint16 |
| BBBC010 brightfield | `bbbc010-brightfield` | uint16 |
| Cell Painting JUMP-Scope | `jump-scope-fluorescence` | uint16 |
| DynaCell A549 phase | `dynacell-a549-phase` | float32 |
| OpenOrganelle / COSEM EM | `cosem-cos7-em` | uint8 |

The OpenCell packs contain twelve 600×600 planes, split evenly between DNA and
tagged-protein fluorescence. Each additional dataset contributes one plane.
Files are headerless, little-endian, C-contiguous plane/Y/X arrays. The manifest
records each asset's type, shape, checksum, and source provenance.
`--dataset opencell-core` retains the original two-pack selection.

On a fresh clone, initialize the submodule, add the annex-bearing Reef checkout
as a read-only Git remote, then retrieve the image assets. Update the URL
if the checkout moves. Reef's noninteractive PATH requires the explicit
`git-annex-shell` path.

```sh
git submodule update --init bench/data/microscopy
git -C bench/data/microscopy remote add reef \
  ssh://login-reef-nclack/mnt/main0/home/nclack/data/chucky-benchmarks
git -C bench/data/microscopy config remote.reef.annex-shell \
  /mnt/main0/home/nclack/.local/share/mamba/envs/git-annex/bin/git-annex-shell
git -C bench/data/microscopy config remote.reef.annex-readonly true
git -C bench/data/microscopy annex get --from=reef data/
python scripts/datasets/run.py verify
uv run scripts/sweep/sweep.py \
  --tier backend --scenario microscopy \
  --build-dir build-gpu --machine reef-l40
```

The GitHub submodule supplies metadata and annex pointers, not the annexed
bytes. Setup and retrieval remain manual; Chucky never fetches content
automatically.

Use `--backend cpu` for a CPU-only build. `export.py` materializes
the selected manifest and image assets.
Each dataset's attribution, license link, and change notice live in the manifest
and therefore travel with every export.
Image loading, padding, timing, profiles, and sweep reporting use the normal
image replay workflow below.

## Cellstate proof of concept

The provisional `cellstate-poc-v1` release has four paired fields from three wells:
four mCherry and four brightfield 2048×2048 uint16 planes at middle Z=3. Each modality
pack is 32 MiB, for 64 MiB total. It preserves stored pixels, while raw acquisition
provenance remains unresolved. The explicit plan keeps all eight images without a
holdout or a claim of representative coverage.

In a saved corpus clone, check out the tag and retrieve
`data/cellstate-poc-v1/` with git-annex, then:

```sh
python scripts/datasets/run.py verify --direct-corpus /path/to/cellstate-poc-v1 \
  --allow-provisional
```

`--allow-provisional` permits only an explicitly provisional manifest with hashed
provenance documenting the missing evidence. Pack and plane checksums remain
mandatory, and this option cannot qualify a source in a raw release. Use this flag
with `export.py` for materialized copies on auk or oreb.

New execution requires an explicit input mapping in `bench/data.json` and a
compatible scenario/input pair in `bench/workloads.toml`; `--direct-corpus` is
verification/export compatibility, not an alternative measurement runner.

Archived summaries report minimum, median, maximum, and range divided by median for
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

For compact manifests, the verifier checks the format declaration, dataset
identity, source-metadata shape, native pixel types, selected asset lengths and
SHA256 values, and the decoded-size cap. Format 1 describes one uint16 dataset;
format 2 describes several datasets with per-asset pixel types. Missing annex content produces a `git annex get data/`
instruction. Git and annex are unnecessary at runtime when materialized files
with the same hashes are available. Evidence-rich schema-1 manifests remain
supported for older corpora.

## Run a pilot

After retrieving the selected annex assets:

```sh
uv run scripts/sweep/sweep.py \
  --tier backend --scenario microscopy \
  --build-dir build-gpu --machine reef-l40
```

Defaults are five measured process executions per input, backend, codec, and
chunk target. Each warms the measured pipeline and streams at least 32 GiB of
logical input, rounded up to whole frames. `--warmup`, `--duration`, and coverage
qualification are shared with generated inputs; retries are not repetitions.
The runner rejects a measurement when its internal clock materially exceeds the
supervising process clock, which catches system sleep during a timed run.
The profiles are none, raw LZ4 level 1, Zstd level 3, Blosc-LZ4 level 3, and
Blosc-Zstd level 3.
Both Blosc profiles use bitshuffle and 16 KiB blocks. The GPU path uses nvCOMP's
default compression modes; its nonzero requested levels are recorded as hints,
including Zstd. CPU levels and GPU modes are not equivalent algorithm settings.

The image scenario uses the regular 16 KiB through 2 MiB chunk matrix, all five
codecs, and both CPU and GPU backends. The six selected inputs therefore produce
480 configurations and 2,400 process executions. `--backend cpu` or `--backend gpu`
filters that to 240 configurations and 1,200 executions. The `compress` and
`backend` tiers select the same image axes; their distinction still applies to
ordinary scenarios. Add `--dry-run` to print the matrix without opening the image
assets. For example, a CPU-only image sweep is:

```sh
uv run scripts/sweep/sweep.py \
  --tier compress --backend cpu --scenario microscopy \
  --build-dir build --machine local
```

Repeat `--scenario` to put images and ordinary scenarios in the same sweep JSON.
`scripts/datasets/run.py run` now forwards to `sweep.py --scenario microscopy`.
It accepts sweep options (`--build-dir`, `--backend`, `--codec`, `--chunk-bytes`)
and writes sweep JSON; the former standalone runner options are no longer used.
`verify` and `compare` remain available for archived standalone corpora/results.

Image chunks preserve the default T:Y:X bit ratio `1:4:4`; every requested
decoded size is checked against the actual layout. All tiers retain the 0.5 GiB
uncompressed full-shard floor, 1 GiB ceiling, 64 MiB batch target, four workers,
one scale, and discard sink. Images retain native dimensions. The replay buffer
is zero-padded to whole chunks before timing; source and padded bytes are
reported separately. Primary throughput counts submitted bytes including this
padding; `throughput_logical_gibs` counts only native image bytes. A memory constraint or unattainable chunk target
causes failure instead of silently changing geometry.

The unified runner writes the normal
`bench/results/<machine>-<commit>-<date>.json`. Each image configuration is one
row containing median throughput, repeat spread, the detailed execution closest
to the median, registered corpus provenance, and the requested chunk identity.
The existing sweep report reads it alongside ordinary scenarios:

```sh
uv run scripts/sweep/report.py --results-dir bench/results/ -o _site --serve
```

The [reporting workflow](../sweep/README.md#microscopy-image-inputs) shows medians,
repeat spread, and stage details. Report input IDs come from `bench/data.json`,
not names or grouping metadata in the data repository.
The JSON includes input and output bytes, actual layouts,
initialization/loading/drain times, memory measurements, image order, corpus
identity, executable hash, source hashes, and machine information.
Compiler, CUDA, and codec-header versions are collected from the executable's
CMake build when available. `environment.py` can also save a build record with
the source content hash.
Archived standalone schema-1 and schema-2 result directories remain reportable.

`logical_compression_fold` is logical image bytes divided by bytes sent to the
sink. The denominator includes sink traffic and its alignment/index overhead.
New sweep rows keep `compression_fold` as decoded full-chunk bytes divided by
physical output, matching generated runs. The logical ratio is separate.
Archived standalone summaries used logical bytes for both fields; their values
are preserved and are not compared automatically across the policy change. The discard sink's
alignment is 4096 bytes.

For a focused functional check (coverage still applies):

```sh
uv run scripts/sweep/sweep.py --tier backend --scenario microscopy --backend cpu \
  --input opencell-dna --codec zstd --chunk-bytes 32K \
  --smoke --min-gib 0.016 --repeats 1 --duration 0.25 \
  --build-dir build --machine local -o /tmp/chucky-images-smoke.json
```
Smoke sweeps are labeled and excluded from the performance trend.

Regular throughput runs use all six registered microscopy assets. This corpus has no
heldout sample, so its throughput and compression measurements are not evidence
of broader representativeness.

## Reuse on auk and oreb

Initialize the submodule and retrieve its selected assets on each machine. The
submodule commit is a reproducible default, but another checkout is accepted when
its manifest contract and selected content verify. The runner records the actual
data commit, manifest hash, and selected asset hashes. The standalone data tools
use Python 3.10 or newer; `sweep.py` declares Python 3.11 or newer. Neither path
needs an image library.

For native Windows with git-annex, use an unlocked adjusted branch:

```powershell
git checkout -b benchmark-opencell data-opencell
git annex adjust --unlock
git annex get data/opencell-v1/
python C:/src/chucky/scripts/datasets/run.py verify --corpus .
```

The adjusted Git HEAD is recorded alongside the manifest and pixel checksums.
[Git-annex documents this branch mode](https://git-annex.branchable.com/git-annex-adjust/).

To transfer normal files instead, export on a compute node after retrieving
annex content:

```sh
python scripts/datasets/export.py --corpus ~/data/chucky-benchmarks \
  --output ~/tmp/chucky-benchmarks-v1.zip
```

Unpack the ZIP on either machine and pass that directory as `--corpus`; this
keeps the registered Chucky selection even if the source manifest lists more
assets than the ZIP contains.
On oreb, set `--build-dir` to the native Windows build directory; the runner
resolves `bench/bench_stream_images.exe`. Set `--machine auk` or `--machine oreb`
and write each sweep to a new output JSON.

Compare archived standalone runs (new sweep results use `report.py`):

```sh
python scripts/datasets/run.py compare \
  bench/results/images/auk-v1/results.json \
  bench/results/images/oreb-v1/results.json \
  --output bench/results/images/auk-oreb-v1.json
```

The comparison refuses different selected input hashes, dataset contracts,
protocols, Chucky source revisions, benchmark source hashes, or actual layouts.
It tolerates data-repository and manifest changes that leave selected inputs
unchanged. Toolchain versions are retained:
different CUDA/nvCOMP versions can affect both compression ratio and throughput,
so such a comparison includes software differences as well as hardware.

## Legacy extraction tools

The extraction pipeline below produces the older evidence-rich manifest format.
It remains available for existing corpora and synthetic tests, but it is not
needed to consume the compact OpenCell corpus.

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
`test_data_sources.py` covers registry validation, version contracts, explicit
logical input mapping, and selection of a subset from a larger repository.
`test_manifest.py` covers compact and legacy formats, corruption, missing
content, invalid manifests, field leakage, and file-copy equivalence.
`test_extract.py` checks exact plane
selection, missing chunks, malformed chunks, float rejection, and full synthetic
extraction. `test_storage.py` checks real annex clone/get/fsck and materialized
export. `test_runner.py` checks delegation to the sweep runner.
`verify_output.py` independently opens saved outputs with Zarr Python and
compares every pixel for native uint8, uint16, and float32 inputs across four
profiles and two append sizes on each backend,
including warmup frames and any extra frames needed for measurement coverage.
It reads bounded blocks so extended runs fit in memory. CTest runs this check
as `test-bench_image_readback_<backend>` when Blosc is available.
With `--corpus`, it also replays the registered assets. Use `--direct-corpus`
and any applicable provenance flag for an older unregistered corpus. In either
case it compares every saved pixel and checks the stored Blosc settings.

```sh
uv run scripts/datasets/verify_output.py \
  --executable build-cpu/bench/bench_stream_images \
  --backends cpu --output /tmp/chucky-image-readback
```

The independent reader follows the
[Zarr Python API](https://zarr.readthedocs.io/en/stable/api/zarr/) and
[Tifffile's page-reading interface](https://github.com/cgohlke/tifffile).

## Data versions and local paths

`bench/data.json` is the only Chucky file that describes logical data sources.
Each `source` gives a stable ID, submodule-relative path, manifest path, and
accepted manifest format version. Each `dataset` gives a stable selection ID,
the data repository's dataset ID and version, and an ordered map from physical
asset IDs to Chucky report input IDs. Multiple datasets can select different
subsets of one source. The default dataset is used unless `--dataset` names
another; `--corpus` changes only the selected source's checkout path.

The submodule supplies a reproducible default revision without making that
revision a runtime requirement. Verification reads only selected content and
checks it against the data repository's manifest. The recorded dataset contract,
actual data commit, whole-manifest hash, selected asset hashes, and direct input
paths preserve provenance while allowing unrelated repository metadata and
unselected assets to change. A future repository rename changes only the URL in
`.gitmodules`; the stable submodule path and IDs in `bench/data.json` stay the same.

The registry's top-level `version` versions its own syntax. A source's
`format_version` is the external manifest/encoding contract, while a dataset's
`manifest_version` is the version published by the data repository. When that
repository intentionally changes either contract, update `bench/data.json` in
the Chucky change that adopts it. Repository commits and metadata-only edits are
recorded as provenance, not treated as version pins.

Selected assets carry SHA256 digests in the source manifest. When git-annex is
available, verification asks it for each asset's content location; otherwise it
uses the resolved file. This works with locked annex links, unlocked files, and
materialized exports without treating an annex key as the data contract.

The verifier fully resolves each selected path before replay. For a locked annex
file this is the object under the real Git directory: `.git/annex/objects/...`
in a normal clone or `.git/modules/.../annex/objects/...` in a submodule.
`pack_path` retains the portable repository-relative name; `resolved_pack_paths`
and each execution's `input_path` show the absolute file actually opened.
Machine-specific paths and data-repository revisions do not enter comparison
identity; selected asset hashes, logical inputs, and replay order do.
