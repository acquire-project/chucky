# Streaming Chunk Management

This document describes how chunks are tracked and their memory managed during
streaming transpose of a D-dimensional tensor.

## Setup

Given input tensor shape $(s_{D-1}, \ldots, s_0)$ and chunk shape
$(n_{D-1}, \ldots, n_0)$, define:

- Chunk counts: $t_d = \lceil s_d / n_d \rceil$
- Lifted shape (row-major): $(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$

Input data arrives in row-major order. In the lifted shape the dimensions from
fastest to slowest varying are:

$$n_0,\; t_0,\; n_1,\; t_1,\; \ldots,\; n_{D-1},\; t_{D-1}$$

## Input strides in the lifted layout

$$
\text{stride}(n_d) = \prod_{k=0}^{d-1}(n_k \cdot t_k)
\qquad
\text{stride}(t_d) = n_d \cdot \prod_{k=0}^{d-1}(n_k \cdot t_k)
$$

## Chunk lifetimes

Each chunk $T = (t_{D-1}, \ldots, t_0)$ receives input elements over a
contiguous range of input indices $[\text{first}(T),\; \text{last}(T)]$:

$$
\text{first}(T) = \sum_{d=0}^{D-1} t_d \cdot \text{stride}(t_d)
$$

$$
\text{last}(T) = \text{first}(T) + C
\qquad\text{where } C = \sum_{d=0}^{D-1} (n_d - 1)\cdot\text{stride}(n_d)
$$

The offset $C$ is the same for every chunk. This means **chunks finish in the
same order they start** (strictly FIFO).

## Epochs

The input partitions into non-overlapping **epochs** indexed by $t_{D-1}$
(the slowest-varying chunk dimension). Epoch $e$ covers input indices:

$$[e \cdot \text{stride}(t_{D-1}),\;\; (e+1) \cdot \text{stride}(t_{D-1}))$$

Within an epoch, the active chunks are all those sharing the same $t_{D-1}$
value. They activate and retire in lexicographic order of
$(t_{D-2}, \ldots, t_0)$.

Epochs do not overlap: the last element of epoch $e$ is at index
$(e+1)\cdot\text{stride}(t_{D-1}) - 1$, and epoch $e+1$ starts at exactly the
next index.

## Peak live chunk count

The maximum number of simultaneously live chunks is:

$$M = \prod_{d=0}^{D-2} t_d$$

This is the product of chunk counts along all dimensions except the
slowest-varying one. It is computable at stream creation time:

```
t_d = ceil(dimensions[d].size / dimensions[d].chunk_size)
M   = product of t_d for d = 0 .. rank-2
```

The peak occurs within each epoch when the earliest chunks have not yet retired
but all chunks in the epoch have been touched. Earlier chunks in the epoch
(smaller lexicographic index) retire before later ones, so the actual live
count drops below $M$ as the epoch progresses.

## Ring buffer

Since chunks are FIFO within each epoch and epochs do not overlap, a **ring
buffer of $M$ chunk buffers** on the GPU suffices. Each buffer holds one full
chunk ($\prod_d n_d$ elements).

```
GPU chunk pool: M slots
┌────────┬────────┬─────┬──────────┐
│ slot 0 │ slot 1 │ ... │ slot M-1 │
└────────┴────────┴─────┴──────────┘
    ↑ tail (oldest active, next to flush)
                    ↑ head (newest active, receiving data)
```

The chunk-to-slot mapping is `flat_chunk_index % M`, where `flat_chunk_index` is
the lexicographic rank of $(t_{D-2}, \ldots, t_0)$ within the current epoch.
By the time a slot would be reused, its previous occupant has retired (the
FIFO property guarantees this).

## Flush detection

Given the current input position `cursor` (in elements), flush the chunk at the
tail whenever:

$$\text{cursor} > \text{last}(\text{tail\_chunk})$$

Since `last(T) = first(T) + C` and chunks retire in order, only the tail needs
to be checked. After flushing, advance the tail and schedule D2H transfer for
that slot.

## Kernel responsibilities

For each input element at global index $i$:

1. Decompose $i$ in the lifted mixed-radix system to get
   $(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$
2. Compute `slot = flat_chunk_index(t_{D-2}, ..., t_0) % M`
3. Compute the intra-chunk offset from $(n_{D-1}, \ldots, n_0)$ as a row-major
   index within the chunk
4. Write the element to `chunk_pool[slot][intra_chunk_offset]`

## Example

Shape $(4, 4, 6)$, chunk $(2, 2, 3)$. So $t = (2, 2, 2)$, $n = (2, 2, 3)$.

Strides: $\text{stride}(n_0)=1$, $\text{stride}(t_0)=3$,
$\text{stride}(n_1)=6$, $\text{stride}(t_1)=12$, $\text{stride}(n_2)=24$,
$\text{stride}(t_2)=48$.

$C = 2 + 6 + 24 = 32$.

$M = t_0 \cdot t_1 = 2 \cdot 2 = 4$ chunk buffers needed.

| Chunk     | first | last | Live range |
|-----------|-------|------|------------|
| (0, 0, 0) |   0   |  32  | [0, 32]    |
| (0, 0, 1) |   3   |  35  | [3, 35]    |
| (0, 1, 0) |  12   |  44  | [12, 44]   |
| (0, 1, 1) |  15   |  47  | [15, 47]   |
| (1, 0, 0) |  48   |  80  | [48, 80]   |
| (1, 0, 1) |  51   |  83  | [51, 83]   |
| (1, 1, 0) |  60   |  92  | [60, 92]   |
| (1, 1, 1) |  63   |  95  | [63, 95]   |

Peak of 4 live chunks occurs at $i \in [15, 32]$. After $i = 32$, chunks begin
retiring and the live count drops until the epoch boundary at $i = 48$.
