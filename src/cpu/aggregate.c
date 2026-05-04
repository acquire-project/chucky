#include "cpu/aggregate.h"

#include "threadpool/threadpool.h"
#include "util/index.ops.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

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

struct gather_ctx
{
  const char* compressed;
  const size_t* comp_sizes;
  const uint32_t* perm;
  const size_t* offsets;
  char* data;
  size_t max_comp;
};

static void
gather_range(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct gather_ctx* c = (struct gather_ctx*)vctx;
  for (size_t i = beg; i < end; ++i) {
    size_t nbytes = c->comp_sizes[i];
    if (nbytes == 0)
      continue;
    const char* src = c->compressed + i * c->max_comp;
    char* dst = c->data + c->offsets[c->perm[i]];
    memcpy(dst, src, nbytes);
  }
}

int
aggregate_cpu_into(const void* compressed,
                   const size_t* comp_sizes,
                   const struct aggregate_layout* layout,
                   struct aggregate_cpu_workspace* ws,
                   struct aggregate_result* result,
                   struct threadpool* pool)
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
    struct gather_ctx c = {
      .compressed = (const char*)compressed,
      .comp_sizes = comp_sizes,
      .perm = ws->perm,
      .offsets = ws->offsets,
      .data = (char*)ws->data,
      .max_comp = max_comp,
    };
    threadpool_for_n(pool, M, gather_range, &c);
  }

  result->data = ws->data;
  result->offsets = ws->offsets;
  result->chunk_sizes = ws->chunk_sizes;
  return 0;
}

struct gather_indirect_ctx
{
  const char* compressed_base;
  const size_t* comp_sizes_base;
  const uint32_t* gather;
  const uint32_t* perm;
  const size_t* offsets;
  char* data;
  size_t max_comp;
};

static void
gather_indirect_range(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct gather_indirect_ctx* c = (struct gather_indirect_ctx*)vctx;
  for (size_t i = beg; i < end; ++i) {
    size_t nbytes = c->comp_sizes_base[c->gather[i]];
    if (nbytes == 0)
      continue;
    const char* src = c->compressed_base + (uint64_t)c->gather[i] * c->max_comp;
    char* dst = c->data + c->offsets[c->perm[i]];
    memcpy(dst, src, nbytes);
  }
}

int
aggregate_cpu_batch_into(const void* compressed_base,
                         const size_t* comp_sizes_base,
                         const uint32_t* gather,
                         const struct aggregate_layout* layout,
                         uint32_t n_active,
                         struct aggregate_cpu_workspace* ws,
                         struct aggregate_result* result,
                         struct threadpool* pool)
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
    struct gather_indirect_ctx c = {
      .compressed_base = (const char*)compressed_base,
      .comp_sizes_base = comp_sizes_base,
      .gather = gather,
      .perm = ws->perm,
      .offsets = ws->offsets,
      .data = (char*)ws->data,
      .max_comp = max_comp,
    };
    threadpool_for_n(pool, batch_M, gather_indirect_range, &c);
  }

  result->data = ws->data;
  result->offsets = ws->offsets;
  result->chunk_sizes = ws->chunk_sizes;
  return 0;
}

// ---- Unified-across-LODs per-batch aggregate ----

int
aggregate_cpu_batch_into_unified(const void* compressed_base,
                                 const size_t* comp_sizes_base,
                                 const uint32_t* gather,
                                 const struct batch_aggregate_layout* layout,
                                 const struct aggregate_layout* per_lod_layouts,
                                 struct shard_state* shards,
                                 size_t* const* h_tail_bytes,
                                 struct aggregate_cpu_workspace* ws,
                                 struct aggregate_result* per_lod_results,
                                 struct threadpool* pool)
{
  const uint64_t total_chunks = layout->total_batch_chunks;
  const uint64_t total_covering = layout->total_batch_covering;
  const uint8_t nlod = layout->nlod;
  const size_t max_comp = layout->max_comp_chunk_bytes;
  const int use_carryover = (layout->page_size > 0);

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

  // Pass 2: per-LOD prefix sum within each LOD's shifted span.
  //
  // Two layouts:
  //  - carry-over (page_size > 0): each shard owns a fixed page-aligned
  //    region of `shard_capacity` bytes. First chunk in shard si is anchored
  //    at `si * shard_capacity + h_tail_bytes[lv][si]` (relative to the LOD
  //    segment start). Chunks pack tightly within the region; the next
  //    batch's leading tail rolls forward via deliver_to_shards_batch.
  //    pad_shard_sizes is intentionally skipped — padding is replaced by
  //    bias-anchored offsets.
  //  - legacy (page_size == 0): contiguous prefix-sum across the LOD's
  //    range, anchored at seg->data_segment_offset (absolute in ws->data).
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    if (seg->n_active == 0)
      continue;
    const uint64_t base = seg->batch_covering_offset + lv;
    const uint64_t cnt = (uint64_t)seg->n_active * seg->covering_count;
    if (use_carryover) {
      const size_t shard_capacity = per_lod_layouts[lv].shard_capacity;
      const uint64_t cps_inner = seg->chunks_per_shard_inner;
      const uint64_t tps_group = (uint64_t)seg->n_active * cps_inner;
      const uint64_t num_shards =
        cps_inner > 0 ? seg->covering_count / cps_inner : 0;
      for (uint64_t si = 0; si < num_shards; ++si) {
        const uint64_t base_si = base + si * tps_group;
        const size_t tail_in =
          (h_tail_bytes && h_tail_bytes[lv]) ? h_tail_bytes[lv][si] : 0;
        ws->offsets[base_si] = (size_t)si * shard_capacity + tail_in;
        for (uint64_t k = 0; k < tps_group; ++k)
          ws->offsets[base_si + k + 1] =
            ws->offsets[base_si + k] + ws->permuted_sizes[base_si + k];
      }
    } else {
      ws->offsets[base] = seg->data_segment_offset;
      for (uint64_t i = 0; i < cnt; ++i)
        ws->offsets[base + i + 1] =
          ws->offsets[base + i] + ws->permuted_sizes[base + i];
    }
  }

  // Leading-tail copy: stage prior batch's ragged tail at the front of each
  // shard's region. CPU equivalent of the GPU's copy_leading_tail_k. Carry-
  // over branch only.
  if (use_carryover && shards) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const struct lod_segment* seg = &layout->lods[lv];
      if (seg->n_active == 0 || !h_tail_bytes || !h_tail_bytes[lv])
        continue;
      const size_t shard_capacity = per_lod_layouts[lv].shard_capacity;
      const struct shard_state* ss = &shards[lv];
      char* seg_base = (char*)ws->data + seg->data_segment_offset;
      for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
        const size_t nbytes = h_tail_bytes[lv][si];
        if (nbytes == 0 || !ss->shards[si].tail_buf)
          continue;
        memcpy(seg_base + (size_t)si * shard_capacity,
               ss->shards[si].tail_buf,
               nbytes);
      }
    }
  }

  // Pass 3: gather compressed chunks. Carry-over offsets are LOD-segment-
  // relative, so each LOD must use its own data base. Legacy offsets are
  // absolute in ws->data; one unified loop suffices.
  if (use_carryover) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const struct lod_segment* seg = &layout->lods[lv];
      if (seg->n_active == 0)
        continue;
      const uint64_t lv_chunks =
        (uint64_t)seg->n_active * seg->chunks_per_epoch;
      if (lv_chunks == 0)
        continue;
      struct gather_indirect_ctx c = {
        .compressed_base = (const char*)compressed_base,
        .comp_sizes_base = comp_sizes_base,
        .gather = gather + seg->batch_chunk_offset,
        .perm = ws->perm + seg->batch_chunk_offset,
        .offsets = ws->offsets,
        .data = (char*)ws->data + seg->data_segment_offset,
        .max_comp = max_comp,
      };
      threadpool_for_n(pool, lv_chunks, gather_indirect_range, &c);
    }
  } else {
    struct gather_indirect_ctx c = {
      .compressed_base = (const char*)compressed_base,
      .comp_sizes_base = comp_sizes_base,
      .gather = gather,
      .perm = ws->perm,
      .offsets = ws->offsets,
      .data = (char*)ws->data,
      .max_comp = max_comp,
    };
    threadpool_for_n(pool, total_chunks, gather_indirect_range, &c);
  }

  // Per-LOD result views. Carry-over uses per-LOD-relative offsets, so
  // result->data is shifted to the LOD segment so deliver-time
  // `result->data + si*shard_capacity + offset_within_shard` is correct.
  // Legacy keeps the unified base.
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    per_lod_results[lv].data =
      use_carryover ? (char*)ws->data + seg->data_segment_offset : ws->data;
    per_lod_results[lv].offsets = ws->offsets + seg->batch_covering_offset + lv;
    per_lod_results[lv].chunk_sizes =
      ws->chunk_sizes + seg->batch_covering_offset + lv;
  }

  return 0;
}
