# Bench layout policy — chunk & shard solve procedure

This document describes how benchmarks (and callers in general) pick chunk sizes
and shard geometry. The solve is structured as two sequential phases because the
problem mostly decouples — see "Why the problem decouples" below.

## Problem

Given an array's shape and dtype, pick:

- `chunk_size[d]` — per-dim chunk size (the unit of compression and GPU transfer).
- `shards[d]` for inner dims — number of shards along each non-append dim.
- `cps_append` — chunks-per-shard along the outer append dim (shard close cadence).

### Inputs

| Variable | Meaning | Source |
|---|---|---|
| `size[d]`, `downsample[d]`, append-prefix | array geometry | caller |
| `bytes_per_element` | dtype size | caller |
| `chunk_ratios[d]` | per-dim chunk-size preference (power-of-2 distribution). `-1` pins chunk_size to full size (for bounded dims; unbounded treated as weight 1). `0` means chunk_size = 1. `>0` = bit-budget weight. | caller |
| `min_chunk_bytes` | chunk-size floor; solve fails if budget can't meet this | caller |
| `target_chunk_bytes` | starting chunk size for the auto-fit loop | caller |
| `memory_max_bytes` | device / heap memory ceiling | caller, or auto-detected |
| `target_concurrent_shards` | preferred number of concurrently-open shards (fs pressure). Soft: the solver aims for this value, but may exceed it when a hard constraint (e.g. `max_parts_per_shard`) requires more inner splitting. A future `max_concurrent_shards` would be the hard ceiling. | caller |
| `min_shard_bytes` | minimum uncompressed bytes per shard (cadence floor) | caller |
| `min_append_shards` | minimum number of shards along the outer append dim; 0 = no minimum. Forces shard-switching in benches that would otherwise collapse to a single shard. | caller |
| `max_parts_per_shard` | backend part-count limit (e.g. S3 multipart = 10000); 0 = unlimited | sink |
| `max_bytes_per_part` | backend per-part byte limit (e.g. S3 = 5 GiB); 0 = unlimited | sink |

### Derived quantities

- `n_chunks[d] = ceildiv(size[d], chunk_size[d])` — chunk count per dim.
- `chunk_bytes = bytes_per_element · Π chunk_size[d]` — bytes per chunk.
- `concurrent_shards = Π shards[d]` for `d ∈ inner` — open-file count at steady state.
- `inner_cps_prod = Π ceildiv(n_chunks[d], shards[d])` for `d ∈ inner` — chunks-per-shard across inner dims.
- `append_row_bytes = chunk_bytes · inner_cps_prod · Π n_chunks[d]` for `d ∈ append, d > 0` — bytes written per append step across one inner shard.
- `shard_size_bytes = cps_append · append_row_bytes` — uncompressed bytes per closed shard.
- `chunks_per_shard_total = cps_append · inner_cps_prod · Π n_chunks[d]` for `d ∈ append, d > 0` — chunk count in one closed shard.
- `device_bytes` — GPU memory estimate for the configuration (from
  `tile_stream_gpu_memory_estimate`).

### Constraints

Hard (feasibility — solver must satisfy or fail with a specific reason):

1. `chunk_bytes ≥ min_chunk_bytes` and `chunk_bytes ≤ max_bytes_per_part` (if nonzero).
2. `device_bytes ≤ memory_max_bytes`.
3. `1 ≤ shards[d] ≤ n_chunks[d]` for inner dims.
4. `cps_append ≥ 1` and `cps_append ≤ n_chunks[0]` when `n_chunks[0] > 0`.
5. `chunks_per_shard_total ≤ max_parts_per_shard` (if nonzero).

Soft (preferences — solver optimizes for these but yields to any hard constraint):

- `shard_size_bytes ≥ min_shard_bytes` — byte floor on closed shards. Ignored when `min_append_shards > 1` (the caller's shard-switching request wins).
- `concurrent_shards ≤ target_concurrent_shards` — concurrency target. The greedy fills up to this value; may be exceeded if a hard constraint (H5 in particular) demands more inner splitting to shrink `inner_cps_prod`. Violations are reported in the diagnostic.

## Why the problem decouples

`device_bytes` is dominated by the chunk pool and per-chunk pipeline state;
shards contribute only small index buffers. Constraint (2) is therefore
effectively a chunk-only constraint.

Shard variables don't appear in constraints (1) or (2) at all. Chunk variables
appear in shard constraints only through `chunk_bytes`, `n_chunks[d]`, and the
product limit (5).

So the optimization is separable: pick chunks first (constraints 1, 2), then
pick shards (constraints 3, 4) from the resulting `n_chunks[d]`. The only
coupling is constraint (5), `chunks_per_shard_total ≤ max_parts_per_shard`,
which is checked at the phase boundary — if violated, we shrink chunks and
retry.

## Lexicographic objective

1. **Maximize `chunk_bytes`** subject to (1) and (2). Larger chunks give better
   compression ratios, lower per-chunk pipeline overhead, and less bookkeeping.
2. **Maximize `concurrent_shards`** subject to (3). Use the fs-concurrency
   budget the caller specified; more inner shards produce finer shard cadence
   without inflating individual shards.
3. **Maximize `cps_append`** subject to `shard_size_bytes ≥ min_shard_bytes`
   (byte floor) and `chunks_per_shard_total ≤ max_parts_per_shard` (parts cap)
   and `cps_append ≤ n_chunks[0]`. Largest shards that stay inside the backend
   parts limit; `min_shard_bytes` acts as a floor, not a target, so shards end
   up as big as the parts cap (and dim extent) allow.

### Tiebreakers

When multiple dims are equally preferred at a greedy step, the choice is
deterministic:

- **Phase 1 bit allocation** (chunks): on equal `bits[i]/ratio[i]`, the
  higher-indexed dim wins (extra bits go to inner/spatial dims, which are
  typically rightmost).
- **Phase 2A ratio greedy** (inner shards): on equal
  `n_chunks[d]/shards[d]`, the lower-indexed dim wins.
- **Phase 2B largest-cps** (parts-budget split): on equal current
  `cps[d]`, the lower-indexed dim wins. Dims where incrementing shards
  does not reduce `cps` (ceildiv roundoff stall) are skipped.

## Procedure

```
# --- Phase 1: chunks + K (auto-fit loop) ---
target = target_chunk_bytes
loop:
    distribute log2(target / bytes_per_element) bits across dims per chunk_ratios
    → sets chunk_size[d] and thus chunk_bytes, n_chunks[d]

    if chunk_bytes > max_bytes_per_part:   halve target, continue

    # K sub-loop: start with auto-derived K (ceildiv(target_batch_chunks,
    # chunks_per_epoch), pow2, clamped to MAX_BATCH_EPOCHS). If device_bytes
    # exceeds memory_max_bytes, halve K (down to 1) and re-estimate. Pools
    # scale linearly in K, so this is the cheapest relief before shrinking
    # chunks. A user-supplied K is authoritative and is not reduced.
    K = auto_K(target_batch_chunks, chunks_per_epoch)
    while device_bytes(K) > memory_max_bytes:
        if K == 1: break    # K-alone can't fit, shrink chunks instead
        K /= 2
    if device_bytes(K) > memory_max_bytes: halve target, continue
    break
    # If target falls below min_chunk_bytes before breaking:
    # ERROR "no chunk size ≥ min_chunk_bytes fits in budget".

# --- Phase 2: shards (closed-form given chunks) ---
shards[d] = 1 for d in inner

# Phase 2A: fill the concurrency target. Integer-greedy, no pow2 waste.
while Π shards[d] < target_concurrent_shards:
    d* = argmax over d in inner where shards[d]+1 ≤ n_chunks[d] and
          shards[d]+1 keeps Π shards[d] ≤ target_concurrent_shards
          of n_chunks[d] / shards[d]
    if no such d*: break
    shards[d*] += 1

# Phase 2B: enforce parts budget. If inner_cps_prod is too big for the parts
# cap given the cps_append target, keep splitting past the target
# (target_concurrent_shards is soft — a preference, not a hard cap).
cps_append_target = ...   # cps_floor if min_shard_bytes > 0,
                           # floor(n_chunks[0]/min_append_shards) if set,
                           # else 1.
while inner_cps_prod · others_prod · cps_append_target > max_parts_per_shard:
    d* = argmax over d in inner where splitting reduces ceildiv(n_chunks[d],
          shards[d]+1) < ceildiv(n_chunks[d], shards[d])
          of current cps[d]
    if no such d*: break     # can't reduce inner_cps_prod further.
    shards[d*] += 1
    refresh cps_append_target (depends on inner_cps_prod in Mode 2)

# If after Phase 2B, inner_cps_prod · others_prod > max_parts_per_shard,
# the config is infeasible at this chunk size: return error PARTS_LIMIT.
# Halving chunks doesn't help — it grows n_chunks, not shrinks it.

# Append cadence: maximize within the parts cap that Phase 2B reserved.
cps_cap   = min(max_parts_per_shard / inner_prod, n_chunks[0] or ∞)
if min_append_shards > 1 and n_chunks[0] > 0:
    cps_cap = min(cps_cap, floor(n_chunks[0] / min_append_shards))
cps_append = cps_cap        # min_shard_bytes is soft: may be unmet if Phase 2B
                             # couldn't reserve enough budget.

# If min_shard_bytes < chunk_bytes at this target, Phase 2 rejects. The outer
# loop halves `target` and retries; this naturally recovers once chunk_bytes
# drops below min_shard_bytes. If the loop exhausts `floor`, the solver
# returns reason = ADVISE_MIN_SHARD_TOO_SMALL.
```

## Worked example

Inputs (roughly the `medfmt_single` bench on a machine with plenty of GPU memory):

- Array: `zcyx` with `size = (100, 1, 10240, 15360)`, `dtype = u16` → `bytes_per_element = 2`
- `chunk_ratios = (1, 0, 4, 4)`, `target_chunk_bytes = 256 KiB`, `min_chunk_bytes = 16 KiB`
- `memory_max_bytes = 8 GiB`
- `target_concurrent_shards = 16`, `min_shard_bytes = 1 GiB`
- Filesystem sink: `max_parts_per_shard = 0`, `max_bytes_per_part = 0` (unlimited)

**Phase 1.** Distribute `log2(256 KiB / 2) = 17` bits across `ratios (1, 0, 4, 4)`.
Bit-greedy (each step goes to the inner dim with the smallest `bits/ratio`,
ties to the higher-indexed dim) yields `bits = (2, 0, 7, 8)`:

```
chunk_size  = (4, 1, 128, 256)
chunk_bytes = 4 · 1 · 128 · 256 · 2 = 262 144   (256 KiB)
n_chunks    = (25, 1, 80, 60)
```

Assume `device_bytes ≤ 8 GiB` holds at this configuration → Phase 1 accepts.

**Phase 2.** `n_append = 1` (only `z`; `chunk_size[z] = 4 ≠ 1`). Inner dims are
`c, y, x`. Start `shards = (·, 1, 1, 1)`, `concurrent_shards = 1`. Double greedily
while `concurrent_shards · 2 ≤ 16`:

| Step | Candidate (largest `n_chunks/shards`) | Action | `concurrent_shards` |
|---|---|---|---|
| 1 | `y: 80/1 = 80` | `shards[y] = 2` | 2 |
| 2 | `x: 60/1 = 60` | `shards[x] = 2` | 4 |
| 3 | `y: 80/2 = 40` | `shards[y] = 4` | 8 |
| 4 | `x: 60/2 = 30` | `shards[x] = 4` | 16 |

(Dim `c` is skipped throughout — `n_chunks[c] = 1` can't be halved.)

```
shards          = (·, c=1, y=4, x=4)     concurrent_shards = 16
inner_cps       = (c=1, y=ceildiv(80,4)=20, x=ceildiv(60,4)=15)
inner_cps_prod  = 1 · 20 · 15 = 300
append_row_bytes = 262 144 · 300 = 78 643 200   (75 MiB)

cps_floor        = ceildiv(1 GiB, 75 MiB) = 14
cps_cap          = min(10000/300 = 33, n_chunks[z] = 25) = 25
cps_append       = max(14, 25) = 25
shard_size_bytes = 25 · 78 643 200 ≈ 1.83 GiB      (above the 1 GiB floor)
```

**Cross-phase check.** `chunks_per_shard_total = 25 · 300 = 7 500 ≤ 10 000`, accepted.

**Final output:**

```
chunk_size         = (4, 1, 128, 256)
chunks_per_shard   = (z=25, c=1, y=20, x=15)   (dim z is the outer append dim)
shard_size_bytes   ≈ 1.83 GiB

# Total shards: ceildiv(25, 25) append-slots × 16 inner = 1 × 16 = 16 shards.
# Total data:   100 · 10240 · 15360 · 2 B ≈ 30 GiB (matches 16 · ~1.8 GiB).
```

## Edge cases

- **`append_row_bytes > min_shard_bytes`** — `cps_append = 1` and the shard is
  exactly `append_row_bytes`, larger than the floor. This is correct:
  `min_shard_bytes` is a floor, not a target.
- **`append_row_bytes · 1 > max_bytes_per_part · max_parts_per_shard`** — a
  single append-row already exceeds the backend's total shard capacity. Phase 1
  retry. If chunks are already at `min_chunk_bytes`, error cleanly and ask the
  caller to raise `target_concurrent_shards` or lower `min_chunk_bytes`.
- **`target_concurrent_shards = 1`** with a large inner grid — one shard covers
  the whole inner volume. Respects the caller's explicit concurrency policy;
  resulting shards may be large.
- **A downsample dim gets `chunk_size = 1`** — allowed (the configuration is
  interpreted as an append-accumulator case, see `dims_n_append` in
  `src/lod/lod_plan.c`). Callers who don't want this should set
  `min_chunk_bytes` high enough to prevent the shrink.
- **Pinning a dim with `chunk_ratios[d] == -1`** — bounded dims pin at
  `size[d]`; unbounded dim 0 (`size == 0`) falls back to weight-1 so it
  absorbs the remaining bit budget. Mixed example: `{1, -1, -1}` on
  `(t=16M, y=16, x=16)` pins y/x at 16 each and lets t soak the chunk
  byte budget.

## Entry points

- `dims_budget_chunk_bytes` (`src/dimension.h`) — Phase 1 primitive:
  distributes bits into `chunk_size[d]` per `chunk_ratios`. Participants are
  clamped to dim extent. Returns non-zero on invalid input (`bpe == 0`,
  `target < bpe`, pinned dims exceed budget).
- `dims_set_shard_geometry` (`src/dimension.h`) — Phase 2 primitive:
  shard geometry given chunks, `min_shard_bytes`, and
  `target_concurrent_shards`. Returns non-zero if `min_shard_bytes` is smaller
  than one chunk.
- `dims_set_layout` (`src/dimension.h`) — convenience wrapper that runs
  both phases from a single `dims_layout_policy` struct.
- `tile_stream_gpu_advise_layout` (`src/stream.gpu.h`) /
  `tile_stream_cpu_advise_layout` (`src/stream.cpu.h`) — combined solve
  with auto-fit loop, memory budget, and cross-phase parts check. Halves
  `epochs_per_batch` (K) before shrinking chunks when the memory budget
  binds; halves the chunk target when either K at floor=1 still overshoots
  or a cross-phase constraint fails; bails below `min_chunk_bytes`. Fills
  an optional `struct advise_layout_diagnostic`:
  - On **failure**, `reason` is one of:
    - `ADVISE_BUDGET_EXCEEDED` — no (chunk, K) pair fit memory budget
    - `ADVISE_PARTS_LIMIT_EXCEEDED` — parts cap infeasible even after Phase B
    - `ADVISE_MIN_SHARD_TOO_SMALL` — `min_shard_bytes < chunk_bytes`
    - `ADVISE_CHUNK_BUDGET_INFEASIBLE` — pinned dims or input rejected the
      chunk target (not recoverable by halving)
    - `ADVISE_INVALID_CONFIG` — caller-level config or shard-geometry rejection
  - On **success**, `reason = ADVISE_OK`. `actual_concurrent_shards` and
    `actual_shard_bytes` let the caller detect soft-constraint
    compromises: `actual_concurrent_shards > target_concurrent_shards`
    means Phase B split past the target to fit the parts budget;
    `actual_shard_bytes < min_shard_bytes` means the floor was not
    reached. `min_append_shards_overrode_min_shard_bytes` is `1` when
    both knobs were set (caller asked for N append shards *and* a byte
    floor) — `min_append_shards` wins per the spec, floor may be unmet.
- Backend constants (`src/defs.limits.h`) — `MAX_PARTS_PER_SHARD` and
  `MAX_BYTES_PER_PART`, applied uniformly across sinks.
- `run_bench` in `bench/bench_util.c` — wires these into the benchmark
  driver and reports auto-fit outcome or a clean failure.
