#include "stream/types.aggregate.h"

#include "platform/platform.h"
#include "stream/layouts.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <string.h>

size_t
agg_pool_bytes(uint64_t chunk_count,
               size_t max_comp_chunk_bytes,
               uint64_t covering_count,
               uint64_t cps_inner,
               size_t page_size)
{
  CHECK_MUL_OVERFLOW(Overflow, chunk_count, max_comp_chunk_bytes, SIZE_MAX);
  size_t bytes = chunk_count * max_comp_chunk_bytes;
  if (page_size > 0 && cps_inner > 0) {
    uint64_t num_shards = covering_count / cps_inner;
    bytes += num_shards * page_size + page_size;
  }
  return bytes;
Overflow:
  return 0;
}

size_t
agg_pool_bytes_layout(const struct aggregate_layout* layout)
{
  if (!layout || layout->num_shards == 0)
    return 0;
  if (layout->shard_capacity == 0) {
    // Layout was computed without a shard_capacity (page_size == 0 path).
    // Fall back to the un-paginated chunk pool size, sized for the worst-case
    // batch (active_count_max epochs).
    uint32_t active = layout->active_count_max ? layout->active_count_max : 1;
    CHECK_MUL_OVERFLOW(
      Overflow, layout->covering_count, layout->max_comp_chunk_bytes, SIZE_MAX);
    size_t per_epoch = layout->covering_count * layout->max_comp_chunk_bytes;
    CHECK_MUL_OVERFLOW(Overflow, per_epoch, (uint64_t)active, SIZE_MAX);
    return per_epoch * active;
  }
  return layout->num_shards * layout->shard_capacity;
Overflow:
  return 0;
}

void
aggregate_batch_luts(const struct aggregate_layout* agg,
                     const struct level_geometry* levels,
                     int lv,
                     uint32_t active_count,
                     const uint32_t* pool_epochs,
                     uint32_t* out_gather,
                     uint32_t* out_perm)
{
  const uint64_t total_chunks = levels->total_chunks;
  const uint64_t M_lv = agg->chunks_per_epoch;
  const uint32_t cps_inner = (uint32_t)agg->cps_inner;
  const uint32_t num_shards = (uint32_t)(agg->covering_count / cps_inner);

  // Output perm maps (epoch, chunk) → position in shard-major order:
  //   [num_shards, active_count, cps_inner]
  // ravel through lifted strides gives shard-grouped position,
  // second ravel inserts the epoch dimension.
  const uint64_t shard_shape[2] = { num_shards, cps_inner };
  const int64_t shard_strides[2] = { (int64_t)active_count * cps_inner, 1 };

  for (uint32_t a = 0; a < active_count; ++a) {
    uint32_t pool_epoch = pool_epochs[a];
    for (uint64_t j = 0; j < M_lv; ++j) {
      uint64_t idx = (uint64_t)a * M_lv + j;
      out_gather[idx] = (uint32_t)(pool_epoch * total_chunks +
                                   levels->level[lv].chunk_offset + j);

      uint64_t perm_pos =
        ravel(agg->lifted_rank, agg->lifted_shape, agg->lifted_strides, j);
      out_perm[idx] =
        (uint32_t)(ravel(2, shard_shape, shard_strides, perm_pos) +
                   a * cps_inner);
    }
  }
}

// Populate a batch_aggregate_layout from per-LOD layouts and the per-LOD
// active counts in this batch. Each LOD's data segment offset is rounded
// up to page alignment so deliver-time write_direct guards still hold.
int
batch_aggregate_layout_init(struct batch_aggregate_layout* out,
                            const struct aggregate_layout* per_lod,
                            const uint32_t* per_lod_n_active,
                            uint8_t nlod,
                            size_t page_size)
{
  CHECK(Error, out);
  CHECK(Error, per_lod);
  CHECK(Error, per_lod_n_active);
  CHECK(Error, nlod >= 1 && nlod <= LOD_MAX_LEVELS);

  memset(out, 0, sizeof(*out));
  out->nlod = nlod;
  out->page_size = page_size;
  out->max_comp_chunk_bytes = per_lod[0].max_comp_chunk_bytes;

  const size_t page_align =
    page_size > 0 ? page_size : platform_page_alignment();

  uint64_t chunk_acc = 0;
  uint64_t covering_acc = 0;
  size_t data_acc = 0;

  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct aggregate_layout* in = &per_lod[lv];
    struct lod_segment* seg = &out->lods[lv];

    // Pool stride is uniform across LODs (see stream/config.c:357 — same
    // max_output_size is passed to every aggregate_layout_compute call).
    CHECK(Error, in->max_comp_chunk_bytes == out->max_comp_chunk_bytes);

    seg->chunks_per_epoch = in->chunks_per_epoch;
    seg->covering_count = in->covering_count;
    seg->chunks_per_shard_inner = in->cps_inner;
    seg->n_active = per_lod_n_active[lv];

    seg->batch_chunk_offset = chunk_acc;
    seg->batch_covering_offset = covering_acc;
    seg->data_segment_offset = data_acc;

    const uint64_t batch_chunks =
      (uint64_t)seg->n_active * seg->chunks_per_epoch;
    const uint64_t batch_cov = (uint64_t)seg->n_active * seg->covering_count;

    chunk_acc += batch_chunks;
    covering_acc += batch_cov;

    // Per-LOD segment capacity — sized via the same agg_pool_bytes math
    // used by the per-LOD path so behavior is identical when nlod == 1.
    size_t seg_bytes = agg_pool_bytes(batch_chunks,
                                      in->max_comp_chunk_bytes,
                                      batch_cov,
                                      seg->n_active * in->cps_inner,
                                      page_size);
    if (seg_bytes == 0 && batch_chunks > 0) {
      log_error("batch_aggregate_layout_init: lod %u agg_pool_bytes overflow "
                "(batch_chunks=%llu, max_comp=%zu)",
                (unsigned)lv,
                (unsigned long long)batch_chunks,
                in->max_comp_chunk_bytes);
      goto Error;
    }

    seg->data_segment_bytes = seg_bytes;

    // Round each segment up to a page boundary so the next LOD's segment
    // base is also page-aligned within the unified buffer.
    if (lv + 1 < nlod) {
      size_t next = data_acc + seg_bytes;
      next = align_up(next, page_align);
      data_acc = next;
    } else {
      data_acc += seg_bytes;
    }
  }

  out->total_batch_chunks = chunk_acc;
  out->total_batch_covering = covering_acc;
  out->total_data_bytes = align_up(data_acc, page_align);

  return 0;

Error:
  if (out)
    memset(out, 0, sizeof(*out));
  return 1;
}

void
aggregate_batch_luts_unified(const struct batch_aggregate_layout* layout,
                             const struct aggregate_layout* per_lod,
                             const struct level_geometry* levels,
                             const uint32_t* const* pool_epochs,
                             uint32_t* out_gather,
                             uint32_t* out_perm)
{
  for (uint8_t lv = 0; lv < layout->nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    if (seg->n_active == 0)
      continue;

    const uint64_t base_chunk = seg->batch_chunk_offset;
    const uint64_t base_cov = seg->batch_covering_offset;
    const uint64_t segment_chunks =
      (uint64_t)seg->n_active * seg->chunks_per_epoch;

    // Reuse the per-LOD builder against this segment's slice of the LUTs;
    // perm targets it produces are local to the LOD's covering range, so
    // shift them by base_cov to live in the unified covering space.
    aggregate_batch_luts(&per_lod[lv],
                         levels,
                         (int)lv,
                         seg->n_active,
                         pool_epochs[lv],
                         out_gather + base_chunk,
                         out_perm + base_chunk);

    // Shift perm targets by base_cov + lv so each LOD's perm range aligns
    // with the same lv-shifted offsets layout written in aggregate's pass
    // 2. Without the +lv, LOD k's perm targets would index into LOD k-1's
    // last offset, off by one for every LOD past the first.
    for (uint64_t i = 0; i < segment_chunks; ++i)
      out_perm[base_chunk + i] += (uint32_t)(base_cov + lv);
  }
}

int
aggregate_layout_compute(struct aggregate_layout* layout,
                         uint8_t rank,
                         uint8_t n_append,
                         const uint64_t* chunk_count,
                         const uint64_t* chunks_per_shard,
                         uint64_t chunks_per_epoch,
                         size_t max_comp_chunk_bytes,
                         uint32_t active_count_max,
                         size_t page_size)
{
  uint64_t shard_count[HALF_MAX_RANK];
  uint64_t eff_cps[HALF_MAX_RANK];
  uint64_t cps_inner = 1;
  uint8_t D;

  CHECK(Error, layout);
  CHECK(Error, rank >= 1);
  CHECK(Error, rank <= HALF_MAX_RANK);
  CHECK(Error, n_append >= 1 && n_append <= rank);
  CHECK(Error, chunk_count);
  CHECK(Error, chunks_per_shard);
  for (int d = n_append; d < rank; ++d)
    CHECK(Error, chunks_per_shard[d] >= 1);

  memset(layout, 0, sizeof(*layout));
  layout->chunks_per_epoch = chunks_per_epoch;
  layout->max_comp_chunk_bytes = max_comp_chunk_bytes;

  D = rank;
  layout->lifted_rank = 2 * (D - n_append);

  // Build lifted shape and strides for dims n_append..D-1
  // lifted_shape[2*k]   = shard_count[d]
  // lifted_shape[2*k+1] = eff_cps[d]
  // Product of shard_count * cps per inner dim. Cannot overflow uint64_t:
  // rank <= HALF_MAX_RANK (8) and each factor is at most ~2^32.
  layout->covering_count = 1;
  for (int d = n_append; d < D; ++d) {
    eff_cps[d] = chunks_per_shard[d];
    shard_count[d] = ceildiv(chunk_count[d], eff_cps[d]);
    int k = d - n_append;
    layout->lifted_shape[2 * k] = shard_count[d];
    layout->lifted_shape[2 * k + 1] = eff_cps[d];
    layout->covering_count *= shard_count[d] * eff_cps[d];
  }
  CHECK(Error, layout->covering_count <= UINT32_MAX);

  // cps_inner = prod(eff_cps[d] for d=n_append..D-1)
  for (int d = n_append; d < D; ++d)
    cps_inner *= eff_cps[d];

  layout->cps_inner = cps_inner;
  layout->num_shards = layout->covering_count / cps_inner;
  layout->active_count_max = active_count_max;
  layout->page_size = page_size;

  // Per-shard reservation in d_aggregated: enough room for the worst-case
  // batch (active_count_max * cps_inner chunks at max_comp_chunk_bytes each)
  // plus one page for the leading tail carried in from the prior batch,
  // rounded up to a page so adjacent shard regions remain page-aligned.
  if (page_size > 0 && active_count_max > 0) {
    size_t worst = (size_t)active_count_max * cps_inner * max_comp_chunk_bytes;
    layout->shard_capacity = align_up(worst + page_size, page_size);
  } else {
    layout->shard_capacity = 0;
  }

  // Shard strides: stride(sc[d]) = prod(sc[j] for j>d) * cps_inner
  {
    uint64_t sc_accum = 1;
    for (int d = D - 1; d >= n_append; --d) {
      int k = d - n_append;
      layout->lifted_strides[2 * k] = (int64_t)(sc_accum * cps_inner);
      sc_accum *= shard_count[d];
    }
  }

  // Within strides: stride(tps[d]) = prod(tps[j] for j>d)
  {
    uint64_t tps_accum = 1;
    for (int d = D - 1; d >= n_append; --d) {
      int k = d - n_append;
      layout->lifted_strides[2 * k + 1] = (int64_t)tps_accum;
      tps_accum *= eff_cps[d];
    }
  }

  return 0;

Error:
  memset(layout, 0, sizeof(*layout));
  return 1;
}
