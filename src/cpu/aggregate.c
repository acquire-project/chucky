#include "aggregate.h"

#include "index.ops.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

int
aggregate_cpu(const void* compressed,
              const size_t* comp_sizes,
              const struct aggregate_layout* layout,
              struct aggregate_result* result)
{
  const uint64_t M = layout->chunks_per_epoch;
  const uint64_t C = layout->covering_count;
  const uint8_t rank = layout->lifted_rank;
  const size_t max_comp = layout->max_comp_chunk_bytes;

  memset(result, 0, sizeof(*result));

  void* data = NULL;
  size_t* permuted_sizes = (size_t*)calloc(C, sizeof(size_t));
  size_t* offsets = (size_t*)malloc((C + 1) * sizeof(size_t));
  size_t* chunk_sizes = (size_t*)calloc(C, sizeof(size_t));
  uint32_t* perm = (uint32_t*)malloc(M * sizeof(uint32_t));
  CHECK(Error, permuted_sizes && offsets && chunk_sizes && perm);

  // Pass 1: permute sizes — ravel each chunk index through the aggregate
  // layout's lifted shape/strides to get shard-order position.
  for (uint64_t i = 0; i < M; ++i) {
    uint32_t pi =
      (uint32_t)ravel(rank, layout->lifted_shape, layout->lifted_strides, i);
    permuted_sizes[pi] = comp_sizes[i];
    perm[i] = pi;
  }

  // Save pre-padding sizes for shard index.
  memcpy(chunk_sizes, permuted_sizes, C * sizeof(size_t));

  // Pass 1.5: pad shard sizes for page alignment.
  if (layout->page_size > 0 && layout->cps_inner > 0) {
    uint64_t num_shards = C / layout->cps_inner;
    for (uint64_t s = 0; s < num_shards; ++s) {
      uint64_t base = s * layout->cps_inner;
      size_t total = 0;
      for (uint64_t j = 0; j < layout->cps_inner; ++j)
        total += permuted_sizes[base + j];
      size_t aligned = align_up(total, layout->page_size);
      size_t padding = aligned - total;
      if (padding > 0)
        permuted_sizes[base + layout->cps_inner - 1] += padding;
    }
  }

  // Pass 2: exclusive prefix sum.
  offsets[0] = 0;
  for (uint64_t i = 0; i < C; ++i)
    offsets[i + 1] = offsets[i] + permuted_sizes[i];

  // Allocate output buffer.
  if (offsets[C] > 0) {
    // Over-allocate by one page_size for alignment-padding read safety
    // (same as GPU path).
    size_t alloc = offsets[C];
    if (layout->page_size > 0)
      alloc += layout->page_size;
    data = malloc(alloc);
    CHECK(Error, data);
  }

  // Pass 3: gather compressed chunks in shard order.
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < M; ++i) {
    size_t nbytes = comp_sizes[i];
    if (nbytes == 0)
      continue;
    const char* src = (const char*)compressed + i * max_comp;
    char* dst = (char*)data + offsets[perm[i]];
    memcpy(dst, src, nbytes);
  }

  free(permuted_sizes);
  free(perm);

  result->data = data;
  result->offsets = offsets;
  result->chunk_sizes = chunk_sizes;
  return 0;

Error:
  free(permuted_sizes);
  free(offsets);
  free(chunk_sizes);
  free(perm);
  free(data);
  return 1;
}

void
aggregate_cpu_result_free(struct aggregate_result* result)
{
  if (!result)
    return;
  free(result->data);
  free(result->offsets);
  free(result->chunk_sizes);
  memset(result, 0, sizeof(*result));
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
  if (layout->page_size > 0 && layout->cps_inner > 0) {
    uint64_t num_shards = C / layout->cps_inner;
    for (uint64_t s = 0; s < num_shards; ++s) {
      uint64_t base = s * layout->cps_inner;
      size_t total = 0;
      for (uint64_t j = 0; j < layout->cps_inner; ++j)
        total += ws->permuted_sizes[base + j];
      size_t aligned = align_up(total, layout->page_size);
      size_t padding = aligned - total;
      if (padding > 0)
        ws->permuted_sizes[base + layout->cps_inner - 1] += padding;
    }
  }

  // Pass 2: exclusive prefix sum.
  ws->offsets[0] = 0;
  for (uint64_t i = 0; i < C; ++i)
    ws->offsets[i + 1] = ws->offsets[i] + ws->permuted_sizes[i];

  // Pass 3: gather compressed chunks in shard order.
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < M; ++i) {
    size_t nbytes = comp_sizes[i];
    if (nbytes == 0)
      continue;
    const char* src = (const char*)compressed + i * max_comp;
    char* dst = (char*)ws->data + ws->offsets[ws->perm[i]];
    memcpy(dst, src, nbytes);
  }

  result->data = ws->data;
  result->offsets = ws->offsets;
  result->chunk_sizes = ws->chunk_sizes;
  return 0;
}
