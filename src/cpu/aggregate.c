#include "cpu/aggregate.h"

#include "platform/platform.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <omp.h>
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
  CHECK(Error,
        ws->perm && ws->permuted_sizes && ws->offsets && ws->chunk_sizes);

  for (uint64_t i = 0; i < M; ++i)
    ws->perm[i] =
      (uint32_t)ravel(rank, layout->lifted_shape, layout->lifted_strides, i);

  ws->data_capacity = agg_pool_bytes(
    M, layout->max_comp_chunk_bytes, C, layout->cps_inner, layout->page_size);
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
                   struct aggregate_result* result,
                   int nthreads)
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
    pad_shard_sizes(
      ws->permuted_sizes, C, layout->cps_inner, layout->page_size);

  // Pass 2: exclusive prefix sum.
  ws->offsets[0] = 0;
  for (uint64_t i = 0; i < C; ++i)
    ws->offsets[i + 1] = ws->offsets[i] + ws->permuted_sizes[i];

  // Pass 3: gather compressed chunks in shard order.
  {
    int i;
#pragma omp parallel for schedule(static) if (M > 64) num_threads(nthreads)
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
                         struct aggregate_result* result,
                         int nthreads)
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
    pad_shard_sizes(ws->permuted_sizes,
                    batch_C,
                    (uint64_t)n_active * layout->cps_inner,
                    layout->page_size);

  // Pass 2: exclusive prefix sum.
  ws->offsets[0] = 0;
  for (uint64_t i = 0; i < batch_C; ++i)
    ws->offsets[i + 1] = ws->offsets[i] + ws->permuted_sizes[i];

  // Pass 3: gather compressed chunks in shard order.
  {
    int i;
#pragma omp parallel for schedule(static) if (batch_M > 64)                    \
  num_threads(nthreads)
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

// ---- Batch-shaped (unified across LODs) workspace + aggregate ----

int
aggregate_cpu_workspace_init_batch(struct aggregate_cpu_workspace* ws,
                                   uint64_t max_total_chunks,
                                   uint64_t max_total_covering,
                                   size_t max_total_data_bytes,
                                   size_t alignment)
{
  memset(ws, 0, sizeof(*ws));
  if (alignment == 0)
    alignment = platform_page_alignment();

  if (max_total_chunks > 0) {
    ws->perm = (uint32_t*)malloc(max_total_chunks * sizeof(uint32_t));
    CHECK(Error, ws->perm);
  }
  if (max_total_covering > 0) {
    // +LOD_MAX_LEVELS slack across permuted_sizes, offsets, and chunk_sizes
    // leaves room for the per-LOD shift used by the unified aggregate so
    // adjacent LODs' spans never collide.
    const uint64_t cov = max_total_covering + LOD_MAX_LEVELS;
    ws->permuted_sizes = (size_t*)calloc(cov, sizeof(size_t));
    ws->offsets = (size_t*)malloc(cov * sizeof(size_t));
    ws->chunk_sizes = (size_t*)calloc(cov, sizeof(size_t));
    CHECK(Error, ws->permuted_sizes && ws->offsets && ws->chunk_sizes);
  }

  ws->data_capacity = align_up(max_total_data_bytes, alignment);
  if (ws->data_capacity > 0) {
    ws->data = platform_aligned_alloc(alignment, ws->data_capacity);
    CHECK(Error, ws->data);
  }

  return 0;

Error:
  aggregate_cpu_workspace_free_batch(ws);
  return 1;
}

void
aggregate_cpu_workspace_free_batch(struct aggregate_cpu_workspace* ws)
{
  if (!ws)
    return;
  free(ws->perm);
  free(ws->permuted_sizes);
  free(ws->offsets);
  free(ws->chunk_sizes);
  platform_aligned_free(ws->data);
  memset(ws, 0, sizeof(*ws));
}

int
aggregate_cpu_batch_into_unified(const void* compressed_base,
                                 const size_t* comp_sizes_base,
                                 const uint32_t* gather,
                                 const uint8_t* source_lod,
                                 const struct batch_aggregate_layout* layout,
                                 struct aggregate_cpu_workspace* ws,
                                 struct aggregate_result* per_lod_results,
                                 int nthreads)
{
  const uint64_t total_chunks = layout->total_batch_chunks;
  const uint64_t total_covering = layout->total_batch_covering;
  const uint8_t nlod = layout->nlod;
  const size_t max_comp = layout->max_comp_chunk_bytes;

  // Each LOD's offsets array has cnt + 1 entries. Adjacent LODs' end/start
  // positions would collide on one shared index of a single packed array.
  // Solution: shift each LOD's view by `lv` so every LOD owns a disjoint
  // span in offsets/permuted_sizes/chunk_sizes. aggregate_batch_luts_unified
  // applies the same shift to perm targets so all references stay in sync.
  // The trailing +nlod slack on each scratch array (allocated by the
  // workspace) covers the shifted span.
  memset(ws->permuted_sizes, 0, (total_covering + nlod) * sizeof(size_t));

  // Pass 1: scatter compressed sizes into permuted-by-shard layout.
  for (uint64_t i = 0; i < total_chunks; ++i)
    ws->permuted_sizes[ws->perm[i]] = comp_sizes_base[gather[i]];

  // Save pre-padding sizes for the shard-index step.
  memcpy(ws->chunk_sizes,
         ws->permuted_sizes,
         (total_covering + nlod) * sizeof(size_t));

  // Pass 1.5: per-LOD shard padding. Each LOD has its own cps_inner and
  // group geometry, so padding stays per-segment.
  if (layout->page_size > 0) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const struct lod_segment* seg = &layout->lods[lv];
      if (seg->n_active == 0 || seg->chunks_per_shard_inner == 0)
        continue;
      pad_shard_sizes(ws->permuted_sizes + seg->batch_covering_offset + lv,
                      (uint64_t)seg->n_active * seg->covering_count,
                      (uint64_t)seg->n_active * seg->chunks_per_shard_inner,
                      layout->page_size);
    }
  }

  // Pass 2: per-LOD prefix sum within each LOD's shifted span. Anchored
  // at the LOD's page-aligned data_segment_offset so absolute byte
  // positions land in the correct slot of the shared data buffer.
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    if (seg->n_active == 0)
      continue;
    const uint64_t base = seg->batch_covering_offset + lv;
    const uint64_t cnt = (uint64_t)seg->n_active * seg->covering_count;
    ws->offsets[base] = seg->data_segment_offset;
    for (uint64_t i = 0; i < cnt; ++i)
      ws->offsets[base + i + 1] =
        ws->offsets[base + i] + ws->permuted_sizes[base + i];
  }

  // Pass 3: unified parallel gather across all LODs. Pool stride is uniform
  // (max_output_size shared across LODs), so source addressing only needs
  // gather[i] * max_comp regardless of source_lod.
  (void)source_lod;
  {
    int i;
#pragma omp parallel for schedule(static) if (total_chunks > 16)               \
  num_threads(nthreads)
    for (i = 0; i < (int)total_chunks; ++i) {
      const size_t nbytes = comp_sizes_base[gather[i]];
      if (nbytes == 0)
        continue;
      const char* src =
        (const char*)compressed_base + (uint64_t)gather[i] * max_comp;
      char* dst = (char*)ws->data + ws->offsets[ws->perm[i]];
      memcpy(dst, src, nbytes);
    }
  }

  // Per-LOD result views: same data base, per-LOD slice of offsets / sizes,
  // each shifted by `lv` to match the disjoint span layout above.
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    per_lod_results[lv].data = ws->data;
    per_lod_results[lv].offsets = ws->offsets + seg->batch_covering_offset + lv;
    per_lod_results[lv].chunk_sizes =
      ws->chunk_sizes + seg->batch_covering_offset + lv;
  }

  return 0;
}
