# Sharding Layer

After the scatter kernel places input elements into chunks, and chunks are compressed,
we need to group compressed chunks by shard and emit complete shards in
[zarr sharding codec][zarr-shard] format.

[zarr-shard]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html

## Overview

The pipeline after this change:

```
input → H2D → scatter → compress → aggregate-by-shard → D2H → shard index bookkeeping → emit
```

The aggregate-by-shard step is a GPU kernel. It reorders compressed chunks into
shard-major order using the same lifted-stride technique that the scatter kernel
uses for elements, but applied to chunk indices instead. The host receives
pre-grouped data and maintains shard indices.

## Chunk-to-Shard Lifting

### Setup

Given an array of rank $D$ with dimensions `dims[0..D-1]` (slowest to fastest),
each dimension has:

- `size`: extent in elements
- `chunk_size`: chunk extent in elements
- `chunks_per_shard`: number of chunks per shard along this dimension

Derived quantities:

$$
\text{chunk\_count}[d] = \lceil \text{size}[d] / \text{chunk\_size}[d] \rceil
$$

$$
\text{shard\_count}[d] = \lceil \text{chunk\_count}[d] / \text{chunks\_per\_shard}[d] \rceil
$$

### Current chunk lifting (recap)

The input element index $i$ is lifted into coordinates
$(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$ where $t_d$ is the chunk index and $n_d$
is the within-chunk index for dimension $d$. The scatter kernel uses lifted
strides to compute the output position in the chunk pool.

The outermost chunk index $t_0$ defines the epoch. Within each epoch, the chunk
pool holds $M = \prod_{d>0} \text{chunk\_count}[d]$ chunks (the "slot count").
Chunks in the pool are indexed in row-major order of $(t_1, \ldots, t_{D-1})$.

### Shard lifting

Each chunk coordinate $t_d$ further lifts to a shard coordinate $s_d$ and a
within-shard coordinate $w_d$:

$$
s_d = t_d \mathbin{/} \text{chunks\_per\_shard}[d], \quad
w_d = t_d \bmod \text{chunks\_per\_shard}[d]
$$

The full lifted coordinate for an element is:

$$
(s_{D-1}, t_{D-1}, n_{D-1}, \ldots, s_0, t_0, n_0)
$$

The output is organized shard-major:

$$
(s_{D-1}, \ldots, s_0, \; t_{D-1}, \ldots, t_0, \; n_{D-1}, \ldots, n_0)
$$

But we don't actually transpose elements to shard layout. Instead, we transpose
*compressed chunks* — reordering the $M$ chunks within an epoch from chunk-major
to shard-major order.

### The shard permutation

Within an epoch, chunk $i$ (flat index in $0\ldots M{-}1$) has chunk coordinates
$(t_1, \ldots, t_{D-1})$ obtained by unraveling $i$ against
$(\text{chunk\_count}[1], \ldots, \text{chunk\_count}[D{-}1])$.

Its shard-major position $P[i]$ is obtained by:

1. Lift: $s_d = t_d / \text{cps}[d]$, $\;w_d = t_d \bmod \text{cps}[d]$ for each $d > 0$
2. Ravel $(s_1, \ldots, s_{D-1}, w_1, \ldots, w_{D-1})$ in row-major order

This permutation groups all chunks of the same shard contiguously. Within each
shard, chunks appear in row-major order of their within-shard coordinates.

**Lifted strides.** The permutation can be computed per-thread using strides, the
same way the scatter kernel computes output element positions:

Define the "shard-lifted shape" for the inner dimensions:

$$
\text{shape} = (
  \text{shard\_count}[1], \;
  \text{chunks\_per\_shard}[1], \;
  \ldots, \;
  \text{shard\_count}[D{-}1], \;
  \text{chunks\_per\_shard}[D{-}1]
)
$$

and the corresponding strides that produce shard-major ordering:

$$
\text{input strides}: \quad
  \sigma_{\text{shard}}[d] = \prod_{j>d} \text{chunk\_count}[j], \quad
  \sigma_{\text{within}}[d] = \prod_{j>d} \text{chunk\_count}[j] / \text{chunks\_per\_shard}[d+1 \ldots]
$$

Actually, it is simpler to think of it as two separate ravel/unravel operations,
or equivalently, to build a stride table for the input ordering and the output
ordering and use the same `coord * stride` dot product:

For input chunk index $i$, the coordinate vector is obtained by unraveling $i$
against the input shape. Each coordinate is then multiplied by the corresponding
output stride and summed to get $P[i]$. The input shape is
$(\text{chunk\_count}[1], \ldots, \text{chunk\_count}[D{-}1])$, and the output
strides are derived from the shard-major shape.

## GPU Aggregation Kernel

After compression, the GPU has $M$ compressed chunks. Chunk $i$ has:
- Data at `d_compressed + i * max_comp_chunk_bytes`
- Actual size `d_comp_sizes[i]`

The aggregation kernel produces:
- `d_aggregated`: compacted bytes in shard-major chunk order
- `d_offsets[0..M]`: byte offset of each chunk in `d_aggregated`

### Algorithm

**Pass 1: Permute sizes.** For each chunk $i$, compute $P[i]$ and write
`permuted_sizes[P[i]] = d_comp_sizes[i]`.

**Pass 2: Prefix sum.** Exclusive scan of `permuted_sizes` →
`d_offsets[0..M]` where `d_offsets[M]` is the total compressed bytes.

**Pass 3: Gather.** For each chunk $i$, copy `d_comp_sizes[i]` bytes from
`d_compressed + i * max_comp_chunk_bytes` to
`d_aggregated + d_offsets[P[i]]`.

This is a segmented compaction: chunks are reordered and packed densely,
eliminating the padding between chunks that `max_comp_chunk_bytes` spacing
introduces.

### Complexity

$M$ is typically hundreds to low thousands (e.g., 768 for the benchmark
configuration). All three passes parallelize trivially over $M$ chunks.
The gather in pass 3 is the most bandwidth-intensive (copies actual compressed
data), but the total volume is just the compressed epoch size.

## D2H Transfer

For variable-size data, the D2H stream first transfers the offsets and actual
chunk sizes. The host uses that metadata to lay out exact transfer spans in a
leased pinned output, then copies only those spans from `d_aggregated`. Fixed
size data can build the same layout without the metadata transfer.

There are always two device aggregate slots. A slot is reusable as soon as its
copy into the leased output completes. Host outputs come from a separate
bounded pool and remain leased until every filesystem command borrowing their
bytes retires.

## Host-Side Shard Accumulation

### Shard epochs

A "shard epoch" spans `chunks_per_shard[0]` epochs along the outermost
dimension. The outer shard coordinate is $s_0 = \lfloor e / \text{cps}[0] \rfloor$
where $e$ is the epoch index ($= t_0$).

At any time, $S_{\text{inner}} = \prod_{d>0} \text{shard\_count}[d]$ shards are
"active" — those sharing the same $s_0$. When $e$ crosses a shard-epoch
boundary (every `cps[0]` epochs), the active shards are complete and can be
emitted.

### Delivery

After D2H completes, the host iterates over the $M$ chunks in shard-major order:

```
chunks_per_shard_inner = prod(chunks_per_shard[d] for d > 0)

for j in 0..M-1:
    shard_idx   = j / chunks_per_shard_inner
    within_inner = j % chunks_per_shard_inner
    within_outer = epoch % chunks_per_shard[0]
    slot = within_outer * chunks_per_shard_inner + within_inner

    size = h_offsets[j+1] - h_offsets[j]
    src  = host_output + host_offsets[j]

    append src (size bytes) to shards[shard_idx].data
    shards[shard_idx].index[2*slot]     = data_write_offset
    shards[shard_idx].index[2*slot + 1] = size
```

Because chunks arrive in shard-major order, the first `chunks_per_shard_inner`
chunks all go to shard 0, the next batch to shard 1, etc. The inner loop over
each shard's chunks occupies a contiguous slice of the leased host output.

### Shard emission

After `chunks_per_shard[0]` epochs:

1. For each active shard $k$ in $0 \ldots S_{\text{inner}} - 1$:
   - Serialize the index: write `chunks_per_shard_total` pairs of `(offset, nbytes)`
     as uint64 little-endian, append 4-byte CRC32C of the index data
   - Call `sink->emit(shard_coord, data, data_bytes, index, index_bytes)`
2. Reset all shard states (clear data, reinitialize index to `0xFFFFFFFFFFFFFFFF`)

### Edge shards

When the array does not evenly divide into shards, edge shards receive fewer
chunks. Unfilled index slots remain at `(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)`,
which the zarr spec defines as "empty chunk" (filled with the fill value on
read). No special logic is needed — this is the default initialization.

The last shard epoch along $d=0$ may also contain fewer than `cps[0]` epochs.
The `flush()` path emits these partial shards.

### Memory

Active memory = $S_{\text{inner}}$ shard states, each with:
- Data buffer: grows dynamically (worst case ≈ `chunks_per_shard_total * max_comp_chunk_bytes`)
- Index: `chunks_per_shard_total * 16` bytes (fixed)
- One shared scratch buffer for index serialization

For a typical configuration (e.g., 768 chunks/epoch, 4×4×4×1 chunks per shard =
64 chunks/shard): $S_{\text{inner}} = 12$ active shards, each accumulating ≤64
compressed chunks. This is modest.

## Zarr Shard Binary Format

A zarr shard on disk is:

```
┌─────────────────────────────────────┐
│ chunk 0 data (variable length)      │
│ chunk 1 data (variable length)      │
│ ...                                 │
│ chunk N-1 data (variable length)    │
├─────────────────────────────────────┤
│ index:                              │
│   (offset_0, nbytes_0) uint64 LE   │
│   (offset_1, nbytes_1) uint64 LE   │
│   ...                               │
│   (offset_{N-1}, nbytes_{N-1})      │
│   crc32c (4 bytes, LE)              │
└─────────────────────────────────────┘
```

Where $N = \text{chunks\_per\_shard\_total}$, offsets are relative to the start
of the shard, and empty chunks have `offset = nbytes = 0xFFFFFFFFFFFFFFFF`.

The `shard_sink` callback receives the data and index portions separately so the
downstream writer can position them appropriately (the index goes at the end by
default per the zarr spec, or at the start if configured).
