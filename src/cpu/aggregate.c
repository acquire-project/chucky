#include "cpu/aggregate.h"

#include "util/index.ops.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

static void
pad_shard_sizes(size_t* sizes, uint64_t C, uint64_t cps_inner, size_t page_size)
{
  uint64_t num_shards = C / cps_inner;
  for (uint64_t s = 0; s < num_shards; ++s) {
    uint64_t base = s * cps_inner;
    size_t total = 0;
    for (uint64_t j = 0; j < cps_inner; ++j)
      total += sizes[base + j];
    size_t aligned = align_up(total, page_size);
    size_t padding = aligned - total;
    if (padding > 0)
      sizes[base + cps_inner - 1] += padding;
  }
}

// ---- Pre-allocated workspace API ----

int
aggregate_cpu_workspace_init(struct aggregate_cpu_workspace* ws,
                             const struct aggregate_layout* layout)
{
  memset(ws, 0, sizeof(*ws));
  const uint64_t M = layout->chunks_per_epoch;
  const uint64_t C = layout->covering_count;
  const uint8_t rank = layout->lifted_rank;

  ws->perm = (uint32_t*)malloc(M * sizeof(uint32_t));
  ws->permuted_sizes = (size_t*)calloc(C, sizeof(size_t));
  ws->offsets = (size_t*)malloc((C + 1) * sizeof(size_t));
  ws->chunk_sizes = (size_t*)calloc(C, sizeof(size_t));
  CHECK(Error, ws->perm && ws->permuted_sizes && ws->offsets && ws->chunk_sizes);

  for (uint64_t i = 0; i < M; ++i)
    ws->perm[i] =
      (uint32_t)ravel(rank, layout->lifted_shape, layout->lifted_strides, i);

  ws->data_capacity =
    agg_pool_bytes(M, layout->max_comp_chunk_bytes,
                   C, layout->cps_inner, layout->page_size);
  if (ws->data_capacity > 0) {
    ws->data = malloc(ws->data_capacity);
    CHECK(Error, ws->data);
  }

  return 0;

Error:
  aggregate_cpu_workspace_free(ws);
  return 1;
}

void
aggregate_cpu_workspace_free(struct aggregate_cpu_workspace* ws)
{
  if (!ws)
    return;
  free(ws->perm);
  free(ws->permuted_sizes);
  free(ws->offsets);
  free(ws->chunk_sizes);
  free(ws->data);
  memset(ws, 0, sizeof(*ws));
}

int
aggregate_cpu_into(const void* compressed,
                   const size_t* comp_sizes,
                   const struct aggregate_layout* layout,
                   struct aggregate_cpu_workspace* ws,
                   struct aggregate_result* result)
{
  const uint64_t M = layout->chunks_per_epoch;
  const uint64_t C = layout->covering_count;
  const size_t max_comp = layout->max_comp_chunk_bytes;

  memset(result, 0, sizeof(*result));

  // Zero scratch.
  memset(ws->permuted_sizes, 0, C * sizeof(size_t));

  // Pass 1: permute sizes using precomputed perm.
  for (uint64_t i = 0; i < M; ++i)
    ws->permuted_sizes[ws->perm[i]] = comp_sizes[i];

  // Save pre-padding sizes for shard index.
  memcpy(ws->chunk_sizes, ws->permuted_sizes, C * sizeof(size_t));

  // Pass 1.5: pad shard sizes for page alignment.
  if (layout->page_size > 0 && layout->cps_inner > 0)
    pad_shard_sizes(ws->permuted_sizes, C, layout->cps_inner, layout->page_size);

  // Pass 2: exclusive prefix sum.
  ws->offsets[0] = 0;
  for (uint64_t i = 0; i < C; ++i)
    ws->offsets[i + 1] = ws->offsets[i] + ws->permuted_sizes[i];

  // Pass 3: gather compressed chunks in shard order.
  {
    int i;
#pragma omp parallel for schedule(static) if(M > 1024)
    for (i = 0; i < (int)M; ++i) {
      size_t nbytes = comp_sizes[i];
      if (nbytes == 0)
        continue;
      const char* src = (const char*)compressed + i * max_comp;
      char* dst = (char*)ws->data + ws->offsets[ws->perm[i]];
      memcpy(dst, src, nbytes);
    }
  }

  result->data = ws->data;
  result->offsets = ws->offsets;
  result->chunk_sizes = ws->chunk_sizes;
  return 0;
}

int
aggregate_cpu_batch_into(const void* compressed_base,
                          const size_t* comp_sizes_base,
                          const uint32_t* gather,
                          const struct aggregate_layout* layout,
                          uint32_t n_active,
                          struct aggregate_cpu_workspace* ws,
                          struct aggregate_result* result)
{
  const uint64_t M = layout->chunks_per_epoch;
  const uint64_t C = layout->covering_count;
  const uint64_t batch_M = (uint64_t)n_active * M;
  const uint64_t batch_C = (uint64_t)n_active * C;
  const size_t max_comp = layout->max_comp_chunk_bytes;

  memset(result, 0, sizeof(*result));

  // Zero scratch.
  memset(ws->permuted_sizes, 0, batch_C * sizeof(size_t));

  // Pass 1: permute sizes using batch gather + perm.
  for (uint64_t i = 0; i < batch_M; ++i)
    ws->permuted_sizes[ws->perm[i]] = comp_sizes_base[gather[i]];

  // Save pre-padding sizes for shard index.
  memcpy(ws->chunk_sizes, ws->permuted_sizes, batch_C * sizeof(size_t));

  // Pass 1.5: pad shard sizes — per-shard with n_active * cps_inner group size.
  if (layout->page_size > 0 && layout->cps_inner > 0)
    pad_shard_sizes(ws->permuted_sizes, batch_C,
                    (uint64_t)n_active * layout->cps_inner, layout->page_size);

  // Pass 2: exclusive prefix sum.
  ws->offsets[0] = 0;
  for (uint64_t i = 0; i < batch_C; ++i)
    ws->offsets[i + 1] = ws->offsets[i] + ws->permuted_sizes[i];

  // Pass 3: gather compressed chunks in shard order.
  {
    int i;
#pragma omp parallel for schedule(static) if(batch_M > 1024)
    for (i = 0; i < (int)batch_M; ++i) {
      size_t nbytes = comp_sizes_base[gather[i]];
      if (nbytes == 0)
        continue;
      const char* src =
        (const char*)compressed_base + (uint64_t)gather[i] * max_comp;
      char* dst = (char*)ws->data + ws->offsets[ws->perm[i]];
      memcpy(dst, src, nbytes);
    }
  }

  result->data = ws->data;
  result->offsets = ws->offsets;
  result->chunk_sizes = ws->chunk_sizes;
  return 0;
}
