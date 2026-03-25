# Design

## The problem

Scientific imaging instruments produce sustained, high-bandwidth streams of
multidimensional data. A light-sheet microscope, for example, generates
2–10 GB/s of 16-bit pixel values organized across time, channel, z, y, and x
dimensions. These streams may run indefinitely — an instrument can acquire for
hours or days, appending along one dimension with no predetermined end.

The storage system must handle this with a fixed resource footprint: bounded
memory, bounded open file handles, and the ability to write completed regions
incrementally without revisiting earlier data. It must also support random
access to rectangular sub-regions for downstream analysis.

[Zarr v3][zarr-v3] addresses this by partitioning the array into independent,
individually addressable **chunks** that can each be compressed and read in
isolation. Zarr's [sharding codec][zarr-shard] groups chunks into **shards** —
single storage objects containing multiple chunks plus an index — reducing the
number of files and amortizing I/O overhead. Chunk sizes balance compression
ratio against random-access granularity; shard sizes balance file count against
write amplification.

[zarr-v3]: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
[zarr-shard]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html

Because the stream is potentially unbounded, the pipeline cannot buffer the
full array before writing. It must chunk and compress data incrementally. The
key observation is that as data arrives in row-major order, only a bounded set
of chunks are active at any time — those sharing the same position along the
outermost (slowest-varying) dimension, the **append dimension**. We call this
set of simultaneously live chunks an **epoch**. The pipeline processes one epoch
at a time, flushes the completed chunks, and reuses the memory. This bounds
the working set regardless of how long the stream runs. (The formal analysis
is in [streaming.md](streaming.md); the mathematical details appear in the
[Approach](#approach) section below.)

During acquisition, scientists need to visualize incoming data in real time —
zooming and panning across a dataset that may already be hundreds of gigabytes.
This requires a **multiscale pyramid**: progressively downsampled copies of the
array at half the resolution along selected dimensions. Viewers like
Neuroglancer read coarse levels for overview and load finer levels on demand,
even while acquisition is still running. The [OME-NGFF][ome-ngff]
specification standardizes how these pyramids are stored alongside zarr arrays
for bioimaging, and is the target output format.

[ome-ngff]: https://ngff.openmicroscopy.org/0.5/

## Approach

We model the data as a **multidimensional array** of rank $D$ with shape
$(s_0, s_1, \ldots, s_{D-1})$, where $s_d$ is the extent along dimension $d$.
Dimensions are ordered slowest-to-fastest: $d = 0$ varies slowest and
$d = D{-}1$ varies fastest. The array is stored in **row-major order** —
elements are laid out in memory such that the last index changes fastest, and
each element is identified by a flat index $i \in [0, \prod_d s_d)$.

### Mixed-radix representation

The pipeline performs several index-space transformations — chunking, storage
reordering, shard aggregation — that look different but share a common
mechanism. Mixed-radix arithmetic gives us a uniform language for all of them.

The coordinates of this array form a bounded integer lattice — the set of all
$(r_0, \ldots, r_{D-1})$ with $0 \le r_d < s_d$. Interpreting coordinates as
**mixed-radix** numbers — where the shape serves as the **radix vector** —
gives each point in the lattice a natural ordering (the row-major order) and a
flat index. More generally, given any radix vector $(b_0, \ldots, b_{D-1})$,
every non-negative integer $i < \prod_d b_d$ has a unique coordinate vector
$(r_0, \ldots, r_{D-1})$ where $0 \le r_d < b_d$. The two conversions are:

- **Unravel.** Recover coordinates from a flat index by repeated division
  against the radix vector: $r_{D-1} = i \bmod b_{D-1}$, then
  $r_{D-2} = \lfloor i / b_{D-1} \rfloor \bmod b_{D-2}$, and so on.

- **Ravel.** Recover a flat index from coordinates using **place values**.
  Each position $d$ has a place value — the product of all faster-varying
  bases: $\sigma_d = \prod_{k=d+1}^{D-1} b_k$. Then
  $i = \sum_d r_d \cdot \sigma_d$.

These place values are more commonly called **strides**. The strides derived
from the array shape — $\sigma_d = \prod_{k=d+1}^{D-1} s_k$ — are the
**natural strides**, and raveling with them recovers the original row-major
index.

Raveling the same coordinates with a different stride vector produces a
different flat index — placing the element at a different memory location. If
that stride vector is itself the natural strides of some target radix vector,
the mapping is an isomorphism: every input position maps to exactly one output
position and vice versa. This is a transposition.

The pipeline's transformations all reduce to unraveling against one shape and
raveling with the strides of another. The scatter kernel, shard
aggregation, and LOD-to-chunks scatter differ only in which radix and stride
tables they use.

### Lifted coordinates and scatter

The central operation is reorganizing a flat row-major stream into chunks. We do
this by **lifting** — replacing the array shape with a finer one that separates
chunk identity from position within a chunk.

Given the array shape $(s_0, \ldots, s_{D-1})$ and chunk shape $(n_0, \ldots,
n_{D-1})$, define the chunk count $t_d = \lceil s_d / n_d \rceil$ for each
dimension. The original rank-$D$ shape is replaced by a rank-$2D$ **lifted
shape**:

$$(t_0, n_0, \; t_1, n_1, \; \ldots, \; t_{D-1}, n_{D-1})$$

Unraveling a flat index against this shape produces coordinates where $t_d$
identifies the chunk along dimension $d$ and $n_d$ is the position within that
chunk. To assemble chunks, we ravel these same coordinates with the strides of
the **chunk-major** shape:

$$(t_0, \ldots, t_{D-1}, \; n_0, \ldots, n_{D-1})$$

The first $D$ coordinates identify the chunk; the last $D$ identify the position
within it. The two shapes share the same elements, just in a different order,
so the mapping is an isomorphism.

A GPU **scatter kernel** implements this: each thread unravels its input index
against the lifted shape, then ravels the coordinates with the chunk-major
strides, writing the element directly to its chunk slot.

**Epochs and bounded memory.** The coordinate $t_0$ — the chunk index along the
append dimension — partitions the stream into **epochs**. Within an epoch, all
chunks share the same $t_0$ value; there are

$$M = \prod_{d=1}^{D-1} t_d$$

chunks (the product over all dimensions except the append dimension). When
assembling chunks, we map $t_0 \to 0$ so that every epoch's chunks land in the
same $M$ pool slots. The pipeline processes one epoch (or a small batch of $K$
epochs), flushes the completed chunks, and reuses the pool — bounding the
working set regardless of stream length.

**Storage order.** The chunk-major strides encode the desired dimension ordering
in the on-disk layout. Changing the storage order (e.g., from `tzcyx` to
`tczyx`) is a different permutation of the same shape, producing different
strides. The scatter kernel itself is unchanged — only the stride table differs.
However, there is a limitation: The append dimension must remain outermost in
the storage order.

### Batching and compression

Once chunks are assembled in GPU memory, they are compressed in a single batched
call using NVIDIA's nvcomp library (zstd or lz4). nvcomp compresses all chunks
simultaneously, but is most efficient when the batch contains many chunks
(1000+). Depending on the configuration, a single epoch may produce relatively
few chunks, so the pipeline accumulates $K$ consecutive epochs into a **batch**
before triggering the compress → aggregate → transfer sequence. $K$ is chosen
so the total chunk count ($K \times M$ times the number of LOD levels) is large
enough for good GPU occupancy.

The **chunk pool** is a contiguous GPU buffer holding all $K \times M$ chunk
slots (plus LOD level slots). Two pools are allocated: while one receives
scatter writes from the current batch, the other drains through compression and
transfer. This double-buffering ensures the scatter and compress stages overlap
completely.

Each chunk is compressed in place into a fixed-size slot bounded by the codec's
worst-case output. Since compressed sizes vary, gaps appear between chunks:

```
   chunk 0          chunk 1          chunk 2          chunk 3
  ┌────────────────┬────────────────┬────────────────┬────────────────┐
  │▓▓▓▓▓░░░░░░░░░░░│▓▓▓▓▓▓▓▓░░░░░░░░│▓▓░░░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓░░░░░░│
  └────────────────┴────────────────┴────────────────┴────────────────┘
   ▓ = compressed data   ░ = unused gap              |◄─ fixed size ─►|
```

These gaps waste both GPU memory and D2H transfer bandwidth, which motivates
the aggregation step.

### Shard aggregation

After compression, chunks sit in the pool in the order they were scattered
into — epoch-major, then chunk-major within each epoch. But shards group chunks
by chunk locality, which is a different order. A GPU **aggregation** kernel
reorders compressed chunks into shard-major order using a three-pass algorithm:

1. **Permute sizes.** Compute the shard-major destination for each chunk and
   write its compressed size to that position.
2. **Prefix sum.** Exclusive scan over the permuted sizes to compute byte
   offsets.
3. **Gather.** Copy each chunk's compressed bytes to its destination offset.

The result is a single contiguous buffer where chunks are grouped by shard and
packed without gaps:

```
  Before (compressed pool — chunk-major order, fixed-size slots):

   chunk 0   chunk 1   chunk 2   chunk 3   chunk 4   chunk 5
  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
  │▓▓▓░░░░░░│▓▓▓▓▓░░░░│▓▓░░░░░░░│▓▓▓▓▓▓░░░│▓▓▓▓░░░░░│▓▓▓▓▓▓▓░░│
  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
   shard A   shard B   shard A   shard B   shard A   shard B

  After (aggregated — shard-major, compacted):

  ┌───┬──┬────┬─────┬──────┬───────┐
  │▓▓▓│▓▓│▓▓▓▓│▓▓▓▓▓│▓▓▓▓▓▓│▓▓▓▓▓▓▓│
  └───┴──┴────┴─────┴──────┴───────┘
  |◄─shard A─►|◄────shard B───────►|
```

Each shard's chunks are contiguous, so the shard can be written to disk with a
single I/O call. A configurable amount of padding is inserted between
shards to align to page boundaries to support efficient I/O downstream.

The permutation is another instance of the unravel/ravel pattern. Each chunk
coordinate $t_d$ is unraveled into a shard index $s_d$ and within-shard
position $w_d$ (using radix $p_d$), then the full coordinate vector is raveled
with shard-major strides. See [sharding.md](sharding.md) for the derivation.

### Multiscale

Each level of the pyramid is produced by $2\times$ reduction along selected
dimensions. We choose $2\times$ blocks for simplicity and because filters with
integral support have desirable properties from a signal-processing
perspective. We also want to support non-separable filters like median and
min/max-suppression — these preserve foreground signal across scales while
rejecting noise, or faithfully downsample segmentation labels. Supporting
non-separable filters rules out factored 1D passes and requires reducing over
the full $2 \times \ldots \times 2$ block at once.

Computing this pyramid during streaming raises several challenges:

- **Selective downsampling.** Not all dimensions should be reduced — a channel
  dimension, for example, must be preserved. The algorithm must distinguish
  downsampled dimensions from batch dimensions.

- **Non-power-of-two shapes.** Array extents are rarely powers of two, but
  hierarchical $2\times$ reduction naturally assumes they are. The algorithm
  must handle arbitrary shapes without introducing artifacts at boundaries.

- **The append dimension spans epochs.** The inner dimensions — all dimensions
  except the append dimension — are fully available within an epoch and can be
  reduced immediately. But the append dimension extends across epochs: a
  $2\times$ reduction at level $l$ requires data from $2^l$ consecutive epochs.
  Buffering all of these is infeasible — for a 256-extent dimension, the
  pyramid has 8 levels, requiring 256 epochs of buffering.

- **Separability.** Because of the epoch constraint, the reduction along the
  append dimension must be computed independently from the inner reduction.
  This factoring is only exact for **separable** reduction methods — those
  where the result is the same whether applied jointly or in independent
  passes.

The design addresses these with two mechanisms: compacted morton order for
efficient inner reduction on the GPU, and a separable fold for the append
dimension.

#### Compacted morton order

Let $D'$ be the number of downsampled dimensions. The pyramid requires
reducing over $2^{D'}$-element blocks at each level. In row-major order, the
elements of a $2 \times \ldots \times 2$ block are not contiguous — they are
scattered across memory, separated by strides. Reducing them requires gathering
from strided locations, which is inefficient on a GPU.

Instead, we reorder elements into **morton order** — a bit-interleaved indexing
where the **morton index** of a coordinate is formed by interleaving the bits
of the downsampled coordinates. Consider a 3×5 array ($D' = 2$):

```
  Row-major order:                Morton order:

     x=0  x=1  x=2  x=3  x=4       x=0  x=1  x=2  x=3  x=4
  y=0  0    1    2    3    4     y=0  0    1    4    5   16
  y=1  5    6    7    8    9     y=1  2    3    6    7   18
  y=2 10   11   12   13   14     y=2  8    9   12   13   24
```

In row-major order, the 2×2 block at top-left is {0, 1, 5, 6} — not
contiguous. In morton order, the same block is {0, 1, 2, 3} — a contiguous
run. Every group of $2^{D'}$ consecutive morton indices forms one reduction
block.

The **morton index** of a coordinate is formed by interleaving the bits of the
downsampled coordinates: if $r_d(k)$ denotes the $k$-th bit of coordinate
$r_d$, then

$$\text{morton}(r) = \ldots\, r_0(k)\, r_1(k) \cdots r_{D'-1}(k) \;\ldots\; r_0(0)\, r_1(0) \cdots r_{D'-1}(0)$$

Reducing each run of $2^{D'}$ elements produces the next coarser level, and
the process repeats on the reduced output. The full pyramid is computed by
successive reduction of contiguous runs — ideal for GPU parallelism.

The complication is that the array shape is rarely a power of two. The morton
indexing covers a $2^p$-sized bounding box, and many of those indices fall
outside the array. In the 3×5 example, the bounding box is 4×8 (32 entries)
but only 15 are in bounds. The **compacted** morton order assigns each
in-bounds element a dense index: the number of in-bounds morton indices that
precede it. This can be computed in $O(p \cdot D')$ time per element, where $p$
is the number of bits needed to cover the largest extent.

```
  Compacted morton order for a 3×5 array:

  Boxes show 2×2 reduction blocks (clipped to array bounds):

     x=0  x=1   x=2  x=3   x=4
     ┌─────────┬─────────┬────┐
  y=0│  0   1  │  4   5  │ 12 │
  y=1│  2   3  │  6   7  │ 13 │
     ├─────────┼─────────┼────┤
  y=2│  8   9  │ 10  11  │ 14 │
     └─────────┴─────────┴────┘

  Linear layout (boxes correspond to the blocks above):

  ┌─────────┬─────────┬─────┬─────┬─────┬────┐
  │ 0 1 2 3 │ 4 5 6 7 │ 8 9 │10 11│12 13│ 14 │
  └─────────┴─────────┴─────┴─────┴─────┴────┘
   full 2×2  full 2×2  ◄── partial blocks ──►
    block     block     (replicate padded)
```

The first two runs of 4 are complete 2×2 blocks. At the boundaries — the
bottom row and the right column — blocks are partial. Replicate padding fills
the missing positions: edge elements are reduced with copies of themselves
rather than with zeros, avoiding darkening artifacts.

**Hierarchical reduction.** Once elements are in compacted morton order, the
pyramid is built by successive reduction of the linear buffer. Each level
reduces contiguous runs, producing a shorter buffer that is itself in compacted
morton order for the coarser shape ($\lceil s_d / 2 \rceil$ per dimension).
Continuing the 3×5 example:

```
  ┌─────────┬─────────┬─────┬─────┬─────┬────┐
  │ 0 1 2 3 │ 4 5 6 7 │ 8 9 │10 11│12 13│ 14 │ L0 (3×5, 15 elements):
  └─────────┴─────────┴─────┴─────┴─────┴────┘  |
  ┌───────────────────────────────┬──────────┐  ▼
  │ 0         1         2    3    │4      5  │ L1 (2×3, 6 elements):
  └───────────────────────────────┴──────────┘  |
  ┌──────────────────────────────────────────┐  ▼
  │ 0                              1         │ L2 (1×2, 2 elements):
  └──────────────────────────────────────────┘
```

Run lengths vary at each level due to boundary effects and are precomputed
from the shape.

**Morton-to-chunks scatter.** The reduced data at each level is still in
compacted morton order, but the downstream compress and aggregate stages expect
chunks. A final scatter step unravels each element's compacted morton index back
to coordinates and ravels with chunk-major strides — the same unravel/ravel
pattern used for L0 scatter, but applied per level against the level's coarser
shape and chunk configuration. Each level's chunks are placed into the shared
chunk pool alongside L0.

#### Separable fold on the append dimension

The pipeline splits the multiscale reduction into two independent phases:

1. **Inner reduction** (within an epoch): the morton-order reduce handles all
   downsampled inner dimensions. This runs every epoch and produces reduced
   data at each level.

2. **Outer fold** (across epochs, along the append dimension): a per-level
   accumulator collects the inner-reduced output. When $2^l$ epochs have
   been accumulated for level $l$, the level emits its reduced chunks and
   resets. This keeps the memory cost proportional to a single epoch regardless
   of pyramid depth.

This factoring constrains the choice of reduction method. Mean, min, and max
are separable — the factored result equals the joint result. Median is not: the
median of inner medians is not in general the joint median. When median is
configured, the pipeline computes it correctly within each phase, but the
composition across phases is an approximation — it can differ from the joint
median by up to the inter-quartile range of the reduction block.

The two phases can use different reduction operators (e.g., mean for the inner
reduction and max for the outer fold), which is useful when the semantics of
the append dimension differ from the inner dimensions.

## Implementation

Data flows through four CUDA streams. Double-buffered pools and event-based
synchronization overlap every stage.

```
                          ┌─────────────────────────────────────────────────┐
 Host input               │                    GPU                          │
 (row-major bytes)        │                                                 │
        │                 │                                                 │
        ▼                 │                                                 │
 ┌──────────────┐         │                                                 │
 │ Staging      ├── H2D ──┤►  d_staging ───────────┐                        │
 │ (pinned, 2×) │  stream │       │                │                        │
 └──────────────┘         │       ▼                ▼                        │
              compute  ┌► │  ┌─────────┐   ┌──────────────────────────────┐ │
               stream  │  │  │ Scatter │   │ LOD (if multiscale)          │ │
                       │  │  │ kernel  │   │                              │ │
                       │  │  │         │   │  gather → reduce → dim0 fold │ │
                       │  │  │         │   │   → morton-to-chunks scatter │ │
                       │  │  └────┬────┘   └──────────────────────────────┘ │
                       │  │       │                │                        │
                       │  │       ▼                ▼                        │
                       │  │  ┌──────────────────────────┐                   │
                       │  │  │ Chunk pool (2× batched)  │                   │
                       │  │  │ L0..Ln LOD chunks        │                   │
                       └► │  └──────────────┬───────────┘                   │
                          │                 │                               │
                          │                 ▼                               │
              compress ┌► │  ┌──────────────────────────┐                   │
               stream  │  │  │ Batch compress           │                   │
                       │  │  │ (nvcomp lz4/zstd)        │                   │
                       │  │  └─────────────┬────────────┘                   │
                       │  │                │                                │
                       │  │                ▼                                │
                       │  │  ┌──────────────────────────┐                   │
                       │  │  │ Aggregate by shard       │                   │
                       │  │  │ (permute, scan, gather)  │                   │
                       └► │  └─────────────┬────────────┘                   │
                          │                │                                │
                          │                ▼                                │
              d2h      ┌► │  ┌──────────────────────────┐                   │
               stream  │  │  │ D2H transfer             │                   │
                       │  │  │ (offsets, then data)     │                   │
                       └► │  └─────────────┬────────────┘                   │
                          │                │                                │
                          └────────────────┼────────────────────────────────┘
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │ Shard delivery│
                                    │ + index build │
                                    └──────┬────────┘
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │ Zarr v3 store │
                                    │ (direct I/O)  │
                                    └───────────────┘
```

### Pipeline stages

**Ingest (H2D stream).** Host data is copied into one of two pinned staging
buffers. When a buffer is full (or the data segment ends), it is transferred to
the GPU asynchronously. The H2D stream waits on the prior scatter to finish
before overwriting the staging area on the device.

**Scatter (compute stream).** Each thread unravels its input index against the
lifted shape and ravels with chunk-major strides, writing the element to
its chunk pool slot. When multiscale is enabled, the raw data is instead copied
linearly for the LOD pipeline, and L0 scatter happens as part of the LOD stage.

**LOD (compute stream).** If multiscale is enabled, each epoch triggers:
1. *Gather* — reorder elements into compacted morton order
2. *Reduce* — apply the reduction operator across $2 \times \ldots \times 2$
   blocks for each level
3. *Dim0 fold* — accumulate into dim0 reduction buffers
4. *Morton-to-chunks* — unravel morton indices and ravel with per-level
   chunk-major strides to scatter reduced data into chunk regions

Each LOD level produces its own set of chunks, interleaved in the same chunk pool
as L0.

**Compress (compress stream).** All chunks in the batch (across all levels) are
compressed in a single nvcomp batch call.

**Aggregate (compress stream).** Compressed chunks are reordered from chunk-major
to shard-major order using the three-pass algorithm described above.

**D2H (D2H stream).** Two-phase transfer: first the offsets array (small), then
the compressed data (sized by the actual compressed output, not the worst-case
bound).

### Event model

The four CUDA streams form a dependency graph through events. Each stage
records an event when it completes; downstream stages wait on that event
before touching the same buffer.

| Producer | Event | Consumer | Guarantees |
|---|---|---|---|
| H2D stream | `t_h2d_end` | compute stream | Staging data is on device before scatter |
| compute stream | `t_scatter_end` | H2D stream | Scatter is done before next H2D overwrites staging |
| compute stream | `pool_events[k]` | compress stream | All $K$ epochs are scattered before compress starts |
| compress stream | `t_aggregate_end` | D2H stream | Aggregated data is ready before transfer |
| D2H stream | `ready` | host | Host can read pinned buffer for shard delivery |

When multiscale is enabled, the LOD pipeline records additional events on
the compute stream (`t_scatter_end`, `t_reduce_end`, `t_dim0_end`) that the
compress stream also waits on before starting.

No stream-wide barriers (`cuStreamSynchronize`) appear in the steady-state
pipeline. The only host-synchronous wait is on the `ready` event before
delivering completed shards to the sink.

### Double buffering

| Resource          | Count | Purpose                                    |
|-------------------|-------|--------------------------------------------|
| Staging buffers   | 2     | Overlap host memcpy with H2D transfer      |
| Chunk pools       | 2     | Overlap scatter with compress+aggregate    |
| Compressed pools  | 2     | Overlap compress with D2H                  |
| Aggregate buffers | 2     | Overlap aggregate with D2H                 |

Synchronization is event-based, not stream-wide: each double-buffered resource
has a pair of events (one per slot) so that writes to slot A and reads from
slot B proceed concurrently without blocking the entire stream.

D2H transfer is two-phase: first the per-chunk offset array (a small,
synchronous transfer so the host can compute shard boundaries), then the
compressed byte data (asynchronous, sized by actual compressed output rather
than worst-case bounds).

### Memory budget

All GPU memory is allocated deterministically at initialization — no dynamic
growth during streaming. The six pools and how they scale:

| Pool | Size | Notes |
|---|---|---|
| Staging (device) | $2 \times \text{buffer\_capacity}$ | Pinned host mirrors of equal size |
| Chunk pool | $2 \times K \times M_{\text{total}} \times \text{chunk\_stride} \times \text{bpe}$ | $M_{\text{total}}$ = sum of chunks across all LOD levels |
| Compressed pool | $2 \times K \times M_{\text{total}} \times \text{max\_output\_size}$ | Worst-case compressed chunk bound |
| Codec workspace | batch pointer arrays + nvcomp temp buffer | Scales with $K \times M_{\text{total}}$ |
| Aggregate | per-level: 2 slots × (offset + size arrays + gather buffer) | Plus permutation LUTs when $K > 1$ |
| LOD | `d_linear` + `d_morton` + per-level shape/stride/LUT arrays + dim0 accumulators | Only allocated when multiscale is enabled |

The dominant terms are the chunk pool and compressed pool, both proportional to
$K \times M_{\text{total}}$. Increasing the chunk size reduces $M$ (fewer chunks
per epoch) at the cost of larger per-chunk buffers. Increasing $K$ improves GPU
occupancy during compression but increases memory proportionally.

`tile_stream_gpu_memory_estimate` computes the exact budget from a
configuration without allocating anything. Call it before committing to verify
that the working set fits in available GPU memory.

### API

#### Configuration

A stream is configured by filling a `tile_stream_configuration`:

| Field | Type / Range | Purpose |
|---|---|---|
| `rank` | 1–32 | Number of dimensions (`MAX_ZARR_RANK`) |
| `dimensions` | `struct dimension[]` | Per-dimension geometry (see below) |
| `dtype` | `enum dtype` | Element type (11 types, 1–8 bytes; see `src/dtype.h`) |
| `buffer_capacity_bytes` | > 0 | H2D staging buffer size (doubled internally) |
| `codec` | none / lz4 / zstd | Compression codec |
| `epochs_per_batch` | 0 or power of 2 | Epochs per batch ($K$); 0 = auto |
| `target_batch_chunks` | > 0 | Target chunk count for auto-$K$ (default 1024) |
| `reduce_method` | mean / median / min / max / max_suppressed / min_suppressed | Inner LOD reduction |
| `dim0_reduce_method` | (same) | Dim0 LOD reduction |
| `shard_alignment` | 0 or page size | Inter-shard padding for direct I/O |
| `metadata_update_interval_s` | ≥ 0 | Seconds between metadata refreshes |

Each `struct dimension` describes one axis:

| Field | Semantics |
|---|---|
| `size` | Extent; 0 = unbounded (append dimension only) |
| `chunk_size` | Chunk extent in this dimension |
| `chunks_per_shard` | Chunks per shard; must be > 0 when `size` is 0 |
| `name` | Optional label (e.g. `"x"`) |
| `downsample` | Include in LOD pyramid (0 or 1) |
| `storage_position` | Position in storage layout; append dimension must be position 0 |
| `axis_type` | OME-NGFF axis type: `dimension_axis_space`, `_time`, `_channel`, `_other` |

#### Writer interface

The caller interacts with the pipeline through a `struct writer` vtable:

- **`append(self, data)`** — push a contiguous `struct slice` of row-major
  elements. Returns a `struct writer_result` with an error code and a `rest`
  slice pointing to any unconsumed input (empty on success).

- **`flush(self)`** — drain any partially filled epochs. Call once at end of
  stream to ensure all data is written.

#### Shard sink interface

`struct shard_sink` is the extension point for storage backends. Any
implementation that provides these four operations can receive output:

- **`open(self, level, shard_index) → shard_writer*`** — return a writer for
  the given shard. The library calls this once per shard per batch.

- **`update_dim0(self, level, dim0_size)`** — called periodically as the
  append dimension grows, allowing the backend to update metadata.

- **`record_fence(self, level) → io_event`** — snapshot the current I/O
  position for backpressure.

- **`wait_fence(self, level, event)`** — block until the I/O subsystem has
  retired past the given fence, preventing the pipeline from outrunning
  storage.

Each `struct shard_writer` returned by `open` provides:

- **`write(self, offset, beg, end)`** — copy data into the shard at the given
  byte offset.
- **`write_direct(self, offset, beg, end)`** — zero-copy variant; the caller
  guarantees the buffer remains valid until the write completes.
- **`finalize(self)`** — mark the shard complete.

#### Memory estimation

`tile_stream_gpu_memory_estimate` takes a `tile_stream_configuration` and
returns a `tile_stream_memory_info` containing total `device_bytes` and
`host_pinned_bytes`, plus a per-component breakdown (staging, chunk pool,
compressed pool, aggregate, LOD, codec workspace) and derived parameters
(`chunks_per_epoch`, `total_chunks`, `epochs_per_batch`). This lets callers
verify resource requirements before allocating.

#### Example

The following shows how these pieces fit together for an unbounded 3D
stream compressed with zstd:

```c
// 1. Describe the dimensions (slowest to fastest).
//    Dimension 0 has size 0, marking it as the append dimension.
struct dimension dims[3] = {
  { .name = "z", .size = 0,   .chunk_size = 64,  .chunks_per_shard = 2 },
  { .name = "y", .size = 512, .chunk_size = 128, .chunks_per_shard = 2 },
  { .name = "x", .size = 512, .chunk_size = 128, .chunks_per_shard = 2 },
};

// 2. Configure the stream.
struct tile_stream_configuration config = {
  .rank               = 3,
  .dimensions         = dims,
  .dtype              = dtype_u16,
  .buffer_capacity_bytes = 4 * 1024 * 1024,
  .codec              = CODEC_ZSTD,
};

// 3. Create the stream and obtain a writer.
struct tile_stream_gpu *s = tile_stream_gpu_create(&config, &my_zarr_sink);
struct writer *w = tile_stream_gpu_writer(s);

// 4. Push data as it arrives — the library handles tiling,
//    compression, and shard delivery internally.
while (acquiring) {
  struct slice frame = get_next_frame();
  writer_append(w, frame);
}

// 5. Flush remaining data and tear down.
writer_flush(w);
tile_stream_gpu_destroy(s);
```

### Zarr store

The library ships a concrete `shard_sink` targeting Zarr v3 stores with the
sharding codec.

**`zarr_sink`** implements `shard_sink` for a single zarr array.
**`zarr_multiscale_sink`** wraps an array of `zarr_sink` instances — one per
LOD level — and manages OME-NGFF group metadata.

**Writer pool.** Each sink maintains a pool of `shard_writer` objects (one per
inner shard index), reused across epochs. This bounds open file
descriptors regardless of how many shards are written over the stream's
lifetime.

**Unbuffered I/O.** When `shard_alignment > 0`, shard data is written with
`O_DIRECT` (Linux) or `FILE_FLAG_NO_BUFFERING` (Windows). Buffers are
page-aligned and inter-shard padding ensures each write begins on a page
boundary. This bypasses the kernel page cache — important at sustained
multi-GB/s write rates where cache pressure would otherwise evict useful
read-side data.

**Dynamic metadata.** `update_dim0` regenerates the zarr array metadata
(`zarr.json`) as the append dimension grows, and for multiscale sinks also
regenerates OME-NGFF group metadata. Metadata is written at a configurable
interval rather than every epoch.

For the zarr shard binary format, see [sharding.md](sharding.md) and the
[zarr sharding codec specification][zarr-shard].

## Related documents

- [streaming.md](streaming.md) — chunk lifetime analysis, FIFO proof, epoch
  derivation
- [sharding.md](sharding.md) — chunk-to-shard lifting, aggregation kernel, zarr
  shard binary format

## Glossary

**Aggregate.** The GPU stage that reorders compressed chunks from chunk-major to
shard-major order, packing them contiguously for single-call shard writes.

**Append dimension.** The outermost (slowest-varying) dimension of the input
array, along which data arrives incrementally. Must be `storage_position` 0.

**Batch.** A group of $K$ consecutive epochs processed together through
compress → aggregate → D2H. Larger batches improve GPU occupancy during
compression.

**Compacted morton order.** A dense reindexing of array elements by
bit-interleaved coordinates, skipping out-of-bounds entries. Groups
$2^{D'}$-element reduction blocks into contiguous runs.

**Epoch.** The set of chunks sharing the same append-dimension chunk index $t_0$.
One epoch's worth of data fills $M$ chunk slots in the pool.

**Inner dimensions.** All dimensions except the append dimension — the
faster-varying dimensions whose full extent is available within a single epoch.

**Lifted shape.** The rank-$2D$ shape $(t_0, n_0, \ldots, t_{D-1}, n_{D-1})$
that separates chunk identity from intra-chunk position.

**LOD level.** One layer of the multiscale pyramid. Level 0 is the original
resolution; level $l$ is reduced by $2^l$ along each downsampled dimension.

**Place values / strides.** The weight of each coordinate position when
converting to a flat index: $\sigma_d = \prod_{k>d} b_k$.

**Radix vector.** The shape interpreted as mixed-radix bases — each entry is
the number of distinct values a coordinate can take.

**Scatter.** A GPU kernel that writes each input element to its chunk-pool
destination by unraveling against the lifted shape and raveling with
chunk-major strides.

**Shard.** A storage object grouping multiple chunks with an appended index.
Corresponds to a single file or object in the zarr store.

**Chunk.** The smallest independently addressable and compressible unit of the
array. Called "chunk" in zarr v3; not to be confused with the zarr outer chunk,
which is a **shard**.

**Chunk pool.** A contiguous GPU buffer holding $K \times M_{\text{total}}$
chunk slots (across all LOD levels), double-buffered.
