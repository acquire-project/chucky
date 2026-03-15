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

The result is a single contiguous buffer where all tiles belonging to the same
shard are adjacent. This means each shard can be written to disk with one I/O
call. When direct I/O is configured, padding is inserted between shards to
align to page boundaries.

The permutation is another instance of the unravel/ravel pattern. Each tile
coordinate $t_d$ is unraveled into a shard index $s_d$ and within-shard
position $w_d$ (using radix $p_d$), then the full coordinate vector is raveled
with shard-major strides. See [sharding.md](sharding.md) for the derivation.

### Multiscale via compacted morton order

The multiscale pyramid requires reducing over $2 \times \ldots \times 2$ blocks
at each level. Not all dimensions participate in downsampling — a channel
dimension, for example, should not be reduced. Let $D'$ be the number of
downsampled dimensions; the remaining dimensions are batch dimensions.

A naive approach would iterate over $2^{D'}$-element blocks explicitly, but
this maps poorly to GPU execution. Instead, we reorder elements into
**compacted morton order** — a bit-interleaved indexing that places each block
in a contiguous run.

The **morton index** of a coordinate is formed by interleaving the bits of the
downsampled coordinates: if $r_d(k)$ denotes the $k$-th bit of coordinate
$r_d$, then

$$\text{morton}(r) = \ldots\, r_0(k)\, r_1(k) \cdots r_{D'-1}(k) \;\ldots\; r_0(0)\, r_1(0) \cdots r_{D'-1}(0)$$

In this order, every consecutive run of $2^{D'}$ elements forms a
$2 \times \ldots \times 2$ block along the downsampled dimensions. Reducing
each run produces the next coarser level, and the process repeats. The result
is a pyramid of levels computed by successive reduction of contiguous runs —
ideal for GPU parallelism.

The complication is that the array shape is not a power of two. A
$2^p$-sized bounding box would contain many out-of-bounds indices. The
**compacted** morton order skips these, producing a dense sequence that covers
only in-bounds elements. Boundary elements are handled by replicate padding:
edge elements are averaged with copies of themselves rather than with zeros, so
there is no darkening artifact at array boundaries.

### Separable fold on the append dimension

When the append dimension ($d = 0$) participates in downsampling, the
multiscale reduction cannot be computed entirely within a single epoch. The
spatial dimensions are fully available each epoch and can be reduced
immediately via the morton-order scheme above. But $d = 0$ extends across
epochs: a $2\times$ reduction at level $l$ requires data from $2^l$ consecutive
epochs.

Buffering all $2^L$ epochs (where $L$ is the pyramid depth) is infeasible — for
a 256-extent dimension, $L = 8$ and $2^L = 256$ epochs. Instead, the pipeline
splits the reduction into two independent phases:

1. **Spatial reduction** (within an epoch): the morton-order reduce handles all
   downsampled dimensions except $d = 0$. This runs every epoch and produces
   spatially reduced data at each level.

2. **Temporal fold** (across epochs along $d = 0$): a per-level accumulator
   collects the spatially reduced output. When $2^l$ epochs have been
   accumulated for level $l$, the level emits its reduced tiles and resets.

This separation has an important consequence for the choice of reduction
method. Because the spatial and temporal reductions are applied independently,
the overall downsampling is only correct when the method is **separable** —
meaning the result is the same whether the reduction is applied jointly across
all dimensions or factored into independent passes. Mean, min, and max are
separable. Median is not: the median of spatial medians is not in general the
joint median. When median is configured, the pipeline computes it correctly
within each phase, but the composition across phases is an approximation.

The two phases can also use different reduction operators (e.g., mean spatially
and max temporally), which is useful when the semantics of the append dimension
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
