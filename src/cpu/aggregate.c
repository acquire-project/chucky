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
aggregate_cpu_batch_into_unified(const struct aggregate_cpu_inputs* in)
{
  const uint64_t total_chunks = in->layout->total_batch_chunks;
  const uint64_t total_covering = in->layout->total_batch_covering;
  const uint8_t nlod = in->layout->nlod;
  const int use_carryover = (in->layout->page_size > 0);

  // Tail-carry callers must supply both arrays; legacy may pass NULL.
  CHECK(Error, !use_carryover || (in->shards_by_lod && in->h_tail_bytes));

  // Each LOD's offsets array has cnt + 1 entries. Adjacent LODs' end/start
  // would collide; shift each LOD's view by `lv` so every LOD owns a disjoint
  // span. aggregate_batch_luts_unified applies the same shift to perm targets.
  memset(in->ws->permuted_sizes, 0, (total_covering + nlod) * sizeof(size_t));

  // Pass 1: scatter compressed sizes into permuted-by-shard layout.
  for (uint64_t i = 0; i < total_chunks; ++i)
    in->ws->permuted_sizes[in->ws->perm[i]] =
      in->comp_sizes_base[in->gather[i]];

  // Save pre-padding sizes for the shard-index step.
  memcpy(in->ws->chunk_sizes,
         in->ws->permuted_sizes,
         (total_covering + nlod) * sizeof(size_t));

  // Pass 2: per-LOD prefix sum.
  //  - carry-over: each shard's first chunk anchored at
  //    `si * shard_capacity + h_tail_bytes[lv][si]` (LOD-segment-relative);
  //    intra-batch gen boundary advances cursor to the next page.
  //  - contiguous: single prefix-sum anchored at seg->data_segment_offset.
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &in->layout->lods[lv];
    if (seg->n_active == 0)
      continue;
    const uint64_t base = seg->batch_covering_offset + lv;
    const uint64_t cnt = (uint64_t)seg->n_active * seg->covering_count;
    if (use_carryover) {
      const size_t shard_capacity = in->per_lod_layouts[lv].shard_capacity;
      const size_t page = in->layout->page_size;
      const uint64_t cps_inner = seg->chunks_per_shard_inner;
      const uint64_t num_shards =
        cps_inner > 0 ? seg->covering_count / cps_inner : 0;
      const struct shard_state* ss = in->shards_by_lod[lv];
      const uint64_t cps_append = ss->chunks_per_shard_append;
      const uint64_t e0 = ss->epoch_in_shard;
      const size_t* lv_tail = in->h_tail_bytes[lv];
      const uint32_t n_active = seg->n_active;
      for (uint64_t si = 0; si < num_shards; ++si) {
        const uint64_t base_si = base + si * (uint64_t)n_active * cps_inner;
        const size_t tail_in = lv_tail ? lv_tail[si] : 0;
        size_t cur = (size_t)si * shard_capacity + tail_in;
        in->ws->offsets[base_si] = cur;
        uint64_t e = e0;
        for (uint32_t ai = 0; ai < n_active; ++ai) {
          for (uint64_t j = 0; j < cps_inner; ++j) {
            const uint64_t k = (uint64_t)ai * cps_inner + j;
            cur += in->ws->permuted_sizes[base_si + k];
            in->ws->offsets[base_si + k + 1] = cur;
          }
          e += 1;
          if (cps_append > 0 && e >= cps_append) {
            e = 0;
            const uint64_t k_next = ((uint64_t)ai + 1) * cps_inner;
            if (ai + 1 < n_active && page > 0) {
              cur = align_up(cur, page);
              in->ws->offsets[base_si + k_next] = cur;
            }
          }
        }
      }
    } else {
      in->ws->offsets[base] = seg->data_segment_offset;
      for (uint64_t i = 0; i < cnt; ++i)
        in->ws->offsets[base + i + 1] =
          in->ws->offsets[base + i] + in->ws->permuted_sizes[base + i];
    }
  }

  // Leading-tail copy: stage prior batch's ragged tail at the front of each
  // shard's region (CPU equivalent of GPU's copy_leading_tail_k).
  if (use_carryover) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const struct lod_segment* seg = &in->layout->lods[lv];
      if (seg->n_active == 0)
        continue;
      const size_t* lv_tail = in->h_tail_bytes[lv];
      if (!lv_tail)
        continue;
      const struct shard_state* ss = in->shards_by_lod[lv];
      const size_t shard_capacity = in->per_lod_layouts[lv].shard_capacity;
      char* seg_base = (char*)in->ws->data + seg->data_segment_offset;
      for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
        const size_t nbytes = lv_tail[si];
        if (nbytes == 0)
          continue;
        memcpy(seg_base + (size_t)si * shard_capacity,
               ss->shards[si].tail_buf,
               nbytes);
      }
    }
  }

  // Pass 3: gather compressed chunks. Carry-over offsets are LOD-segment-
  // relative; legacy offsets are absolute in ws->data.
  if (use_carryover) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      const struct lod_segment* seg = &in->layout->lods[lv];
      if (seg->n_active == 0)
        continue;
      const uint64_t lv_chunks =
        (uint64_t)seg->n_active * seg->chunks_per_epoch;
      if (lv_chunks == 0)
        continue;
      struct gather_indirect_ctx c = {
        .compressed_base = (const char*)in->compressed_base,
        .comp_sizes_base = in->comp_sizes_base,
        .gather = in->gather + seg->batch_chunk_offset,
        .perm = in->ws->perm + seg->batch_chunk_offset,
        .offsets = in->ws->offsets,
        .data = (char*)in->ws->data + seg->data_segment_offset,
        .max_comp = in->layout->max_comp_chunk_bytes,
      };
      threadpool_for_n(in->pool, lv_chunks, gather_indirect_range, &c);
    }
  } else {
    struct gather_indirect_ctx c = {
      .compressed_base = (const char*)in->compressed_base,
      .comp_sizes_base = in->comp_sizes_base,
      .gather = in->gather,
      .perm = in->ws->perm,
      .offsets = in->ws->offsets,
      .data = (char*)in->ws->data,
      .max_comp = in->layout->max_comp_chunk_bytes,
    };
    threadpool_for_n(in->pool, total_chunks, gather_indirect_range, &c);
  }

  // Per-LOD result views. Carry-over shifts result->data to the LOD segment so
  // deliver-time addressing through `data + si*shard_capacity + offset` is
  // correct; legacy keeps the unified base.
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct lod_segment* seg = &in->layout->lods[lv];
    in->per_lod_results[lv].data =
      use_carryover ? (char*)in->ws->data + seg->data_segment_offset
                    : in->ws->data;
    in->per_lod_results[lv].offsets =
      in->ws->offsets + seg->batch_covering_offset + lv;
    in->per_lod_results[lv].chunk_sizes =
      in->ws->chunk_sizes + seg->batch_covering_offset + lv;
  }

  return 0;

Error:
  return 1;
}
