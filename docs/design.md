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
isolation. (We use the term **tile** for chunk throughout this document to avoid
overloading "chunk," which is used in other contexts.) Zarr's
[sharding codec][zarr-shard] groups tiles into **shards** — single storage
objects containing multiple tiles plus an index — reducing the number of files
and amortizing I/O overhead. Tile sizes balance compression ratio against
random-access granularity; shard sizes balance file count against write
amplification.

[zarr-v3]: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
[zarr-shard]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html

Because the stream is potentially unbounded, the pipeline cannot buffer the
full array before writing. It must tile and compress data incrementally. The
key observation is that as data arrives in row-major order, only a bounded set
of tiles are active at any time — those sharing the same position along the
outermost (slowest-varying) dimension, the **append dimension**. We call this
set of simultaneously live tiles an **epoch**. The pipeline processes one epoch
at a time, flushes the completed tiles, and reuses the memory. This bounds
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

The pipeline performs several index-space transformations — tiling, storage
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
aggregation, and LOD-to-tiles scatter differ only in which radix and stride
tables they use.

### Lifted coordinates and scatter

The central operation is reorganizing a flat row-major stream into tiles. We do
this by **lifting** — replacing the array shape with a finer one that separates
tile identity from position within a tile.

Given the array shape $(s_0, \ldots, s_{D-1})$ and tile shape $(n_0, \ldots,
n_{D-1})$, define the tile count $t_d = \lceil s_d / n_d \rceil$ for each
dimension. The original rank-$D$ shape is replaced by a rank-$2D$ **lifted
shape**:

$$(t_0, n_0, \; t_1, n_1, \; \ldots, \; t_{D-1}, n_{D-1})$$

Unraveling a flat index against this shape produces coordinates where $t_d$
identifies the tile along dimension $d$ and $n_d$ is the position within that
tile. To assemble tiles, we ravel these same coordinates with the strides of
the **tile-major** shape:

$$(t_0, \ldots, t_{D-1}, \; n_0, \ldots, n_{D-1})$$

The first $D$ coordinates identify the tile; the last $D$ identify the position
within it. The two shapes share the same elements, just in a different order,
so the mapping is an isomorphism.

A GPU **scatter kernel** implements this: each thread unravels its input index
against the lifted shape, then ravels the coordinates with the tile-major
strides, writing the element directly to its tile slot.

**Epochs and bounded memory.** The coordinate $t_0$ — the tile index along the
append dimension — partitions the stream into **epochs**. Within an epoch, all
tiles share the same $t_0$ value; there are

$$M = \prod_{d=1}^{D-1} t_d$$

tiles (the product over all dimensions except the append dimension). When
assembling tiles, we map $t_0 \to 0$ so that every epoch's tiles land in the
same $M$ pool slots. The pipeline processes one epoch (or a small batch of $K$
epochs), flushes the completed tiles, and reuses the pool — bounding the
working set regardless of stream length.

**Storage order.** The tile-major strides encode the desired dimension ordering
in the on-disk layout. Changing the storage order (e.g., from `tzcyx` to
`tczyx`) is a different permutation of the same shape, producing different
strides. The scatter kernel itself is unchanged — only the stride table differs.
However, there is a limitation: The append dimension must remain outermost in
the storage order.

### Batching and compression

Once tiles are assembled in GPU memory, they are compressed in a single batched
call using NVIDIA's nvcomp library (zstd or lz4). nvcomp compresses all tiles
simultaneously, but is most efficient when the batch contains many tiles
(1000+). Depending on the configuration, a single epoch may produce relatively
few tiles, so the pipeline accumulates $K$ consecutive epochs into a **batch**
before triggering the compress → aggregate → transfer sequence. $K$ is chosen
so the total tile count ($K \times M$ times the number of LOD levels) is large
enough for good GPU occupancy.

The **tile pool** is a contiguous GPU buffer holding all $K \times M$ tile
slots (plus LOD level slots). Two pools are allocated: while one receives
scatter writes from the current batch, the other drains through compression and
transfer. This double-buffering ensures the scatter and compress stages overlap
completely.

Each tile is compressed in place into a fixed-size slot bounded by the codec's
worst-case output. Since compressed sizes vary, gaps appear between tiles:

```
   tile 0           tile 1           tile 2           tile 3
  ┌────────────────┬────────────────┬────────────────┬────────────────┐
  │▓▓▓▓▓░░░░░░░░░░░│▓▓▓▓▓▓▓▓░░░░░░░░│▓▓░░░░░░░░░░░░░░│▓▓▓▓▓▓▓▓▓▓░░░░░░│
  └────────────────┴────────────────┴────────────────┴────────────────┘
   ▓ = compressed data   ░ = unused gap              |◄─ fixed size ─►|
```

These gaps waste both GPU memory and D2H transfer bandwidth, which motivates
the aggregation step.

### Shard aggregation

After compression, tiles sit in the pool in the order they were scattered
into — epoch-major, then tile-major within each epoch. But shards group tiles
by spatial locality, which is a different order. A GPU **aggregation** kernel
reorders compressed tiles into shard-major order using a three-pass algorithm:

1. **Permute sizes.** Compute the shard-major destination for each tile and
   write its compressed size to that position.
2. **Prefix sum.** Exclusive scan over the permuted sizes to compute byte
   offsets.
3. **Gather.** Copy each tile's compressed bytes to its destination offset.

The result is a single contiguous buffer where tiles are grouped by shard and
packed without gaps:

```
  Before (compressed pool — tile-major order, fixed-size slots):

   tile 0    tile 1    tile 2    tile 3    tile 4    tile 5
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

Each shard's tiles are contiguous, so the shard can be written to disk with a
single I/O call. A configurable amount of padding is inserted between
shards to align to page boundaries to support efficient I/O downstream.

The permutation is another instance of the unravel/ravel pattern. Each tile
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

- **The append dimension spans epochs.** The spatial dimensions within an epoch
  are fully available and can be reduced immediately. But the append dimension
  extends across epochs: a $2\times$ reduction at level $l$ requires data from
  $2^l$ consecutive epochs. Buffering all of these is infeasible — for a
  256-extent dimension, the pyramid has 8 levels, requiring 256 epochs of
  buffering.

- **Separability.** Because of the epoch constraint, the reduction along the
  append dimension must be computed independently from the spatial reduction.
  This factoring is only exact for **separable** reduction methods — those
  where the result is the same whether applied jointly or in independent
  passes.

The design addresses these with two mechanisms: compacted morton order for
efficient spatial reduction on the GPU, and a separable fold for the append
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
  L0 (3×5, 15 elements):
  ┌─────────┬─────────┬─────┬─────┬─────┬────┐
  │ 0 1 2 3 │ 4 5 6 7 │ 8 9 │10 11│12 13│ 14 │
  └────┬────┴────┬────┴──┬──┴──┬──┴──┬──┴─┬──┘
       ▼         ▼       ▼     ▼     ▼    ▼
  L1 (2×3, 6 elements):
  ┌─────────┬─────┬────┐
  │ 0 1 2 3 │ 4 5 │(p) │
  └────┬────┴──┬──┴────┘
       ▼       ▼
  L2 (1×2, 2 elements):
  ┌─────┐
  │ 0 1 │
  └──┬──┘
     ▼
  L3 (1×1, 1 element):
  ┌───┐
  │ 0 │
  └───┘

  (p) = partial block, replicate padded
```

Run lengths vary at each level due to boundary effects and are precomputed
from the shape.

**Morton-to-tiles scatter.** The reduced data at each level is still in
compacted morton order, but the downstream compress and aggregate stages expect
tiles. A final scatter step unravels each element's compacted morton index back
to coordinates and ravels with tile-major strides — the same unravel/ravel
pattern used for L0 scatter, but applied per level against the level's coarser
shape and tile configuration. Each level's tiles are placed into the shared
tile pool alongside L0.

#### Separable fold on the append dimension

The pipeline splits the multiscale reduction into two independent phases:

1. **Spatial reduction** (within an epoch): the morton-order reduce handles all
   downsampled dimensions except the append dimension. This runs every epoch
   and produces spatially reduced data at each level.

2. **Temporal fold** (across epochs along the append dimension): a per-level
   accumulator collects the spatially reduced output. When $2^l$ epochs have
   been accumulated for level $l$, the level emits its reduced tiles and
   resets. This keeps the memory cost proportional to a single epoch regardless
   of pyramid depth.

This factoring constrains the choice of reduction method. Mean, min, and max
are separable — the factored result equals the joint result. Median is not: the
median of spatial medians is not in general the joint median. When median is
configured, the pipeline computes it correctly within each phase, but the
composition across phases is an approximation.

The two phases can use different reduction operators (e.g., mean spatially and
max temporally), which is useful when the semantics of the append dimension
differ from the spatial dimensions.

## Pipeline

Data flows through four CUDA streams. Double-buffered pools and event-based
synchronization overlap every stage.

```
                          ┌─────────────────────────────────────────────────┐
 Host input               │                    GPU                          │
 (row-major bytes)        │                                                 │
        │                 │                                                 │
        ▼                 │                                                 │
 ┌──────────────┐         │                                                 │
 │ Staging      │ ── H2D ─┤►  d_staging                                     │
 │ (pinned, 2×) │  stream │       │                                         │
 └──────────────┘         │       ▼                                         │
                          │  ┌─────────┐   ┌──────────────────────────────┐ │
                          │  │ Scatter │   │ LOD (if multiscale)          │ │
                 compute  │  │ kernel  │   │                              │ │
                  stream  │  │         │──►│  gather → reduce → dim0 fold │ │
                          │  │         │   │   → morton-to-tiles scatter  │ │
                          │  └────┬────┘   └──────────────┬───────────────┘ │
                          │       │                       │                 │
                          │       ▼                       ▼                 │
                          │  ┌──────────────────────────────┐               │
                          │  │ Tile pool (2× batched)       │               │
                          │  │ L0 tiles + L1..Ln LOD tiles  │               │
                          │  └──────────────┬───────────────┘               │
                          │                 │                               │
                 compress │                 ▼                               │
                  stream  │  ┌──────────────────────────┐                   │
                          │  │ Batch compress           │                   │
                          │  │ (nvcomp lz4/zstd)        │                   │
                          │  └─────────────┬────────────┘                   │
                          │                │                                │
                          │                ▼                                │
                          │  ┌──────────────────────────┐                   │
                          │  │ Aggregate by shard       │                   │
                          │  │ (permute, scan, gather)  │                   │
                          │  └─────────────┬────────────┘                   │
                          │                │                                │
                     d2h  │                ▼                                │
                  stream  │  ┌──────────────────────────┐                   │
                          │  │ D2H transfer             │                   │
                          │  │ (offsets, then data)     │                   │
                          │  └─────────────┬────────────┘                   │
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

### Stage details

**Ingest (H2D stream).** Host data is copied into one of two pinned staging
buffers. When a buffer is full (or the data chunk ends), it is transferred to
the GPU asynchronously. The H2D stream waits on the prior scatter to finish
before overwriting the staging area on the device.

**Scatter (compute stream).** Each thread unravels its input index against the
lifted shape and ravels with tile-major strides, writing the element to
its tile pool slot. When multiscale is enabled, the raw data is instead copied
linearly for the LOD pipeline, and L0 scatter happens as part of the LOD stage.

**LOD (compute stream).** If multiscale is enabled, each epoch triggers:
1. *Gather* — reorder elements into compacted morton order
2. *Reduce* — apply the reduction operator across $2 \times \ldots \times 2$
   blocks for each level
3. *Dim0 fold* — accumulate into temporal reduction buffers
4. *Morton-to-tiles* — unravel morton indices and ravel with per-level
   tile-major strides to scatter reduced data into tile regions

Each LOD level produces its own set of tiles, interleaved in the same tile pool
as L0.

**Compress (compress stream).** All tiles in the batch (across all levels) are
compressed in a single nvcomp batch call.

**Aggregate (compress stream).** Compressed tiles are reordered from tile-major
to shard-major order using the three-pass algorithm described above.

**D2H (D2H stream).** Two-phase transfer: first the offsets array (small), then
the compressed data (sized by the actual compressed output, not the worst-case
bound).

**Shard delivery (host).** The host iterates over tiles in shard-major order,
dispatching contiguous runs to per-shard writers and building the shard index.
When a shard's tiles are complete, the index is serialized with a CRC32C
checksum and emitted.

### Double buffering

| Resource          | Count | Purpose                                    |
|-------------------|-------|--------------------------------------------|
| Staging buffers   | 2     | Overlap host memcpy with H2D transfer      |
| Tile pools        | 2     | Overlap scatter with compress+aggregate    |
| Compressed pools  | 2     | Overlap compress with D2H                  |
| Aggregate buffers | 2     | Overlap aggregate with D2H                 |

Event-based synchronization (no stream-wide barriers) ensures each stage waits
only on its actual data dependency.

## Related documents

- [streaming.md](streaming.md) — tile lifetime analysis, FIFO proof, epoch
  derivation
- [sharding.md](sharding.md) — tile-to-shard lifting, aggregation kernel, zarr
  shard binary format
