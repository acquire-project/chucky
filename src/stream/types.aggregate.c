#include "stream/types.aggregate.h"

#include "stream/layouts.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <string.h>

size_t
footer_capacity_for(uint64_t chunks_per_shard_total, size_t page_size)
{
  if (page_size == 0)
    return 0;
  // Footer layout: [<page trailing data || index || CRC || pad-to-page].
  // Worst-case total = (page-1) + chunks*16 + 4, rounded up to page.
  CHECK_MUL_OVERFLOW(Overflow, chunks_per_shard_total, (uint64_t)16, SIZE_MAX);
  size_t index_bytes = (size_t)chunks_per_shard_total * 16;
  if (index_bytes > SIZE_MAX - 4)
    return 0;
  size_t logical = index_bytes + 4;
  if (logical > SIZE_MAX - page_size)
    return 0;
  return align_up(page_size + logical, page_size);
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

int
batch_aggregate_layout_init_compact(struct batch_aggregate_layout* out,
                                    const struct aggregate_layout* per_lod,
                                    const uint32_t* per_lod_n_active,
                                    uint8_t nlod)
{
  CHECK(Error, out && per_lod && per_lod_n_active);
  CHECK(Error, nlod >= 1 && nlod <= LOD_MAX_LEVELS);

  memset(out, 0, sizeof(*out));
  out->nlod = nlod;
  out->max_comp_chunk_bytes = per_lod[0].max_comp_chunk_bytes;

  uint64_t chunk_acc = 0;
  uint64_t covering_acc = 0;
  size_t data_acc = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct aggregate_layout* in = &per_lod[lv];
    struct lod_segment* seg = &out->lods[lv];
    CHECK(Error, in->max_comp_chunk_bytes == out->max_comp_chunk_bytes);
    CHECK_MUL_OVERFLOW(
      Error, per_lod_n_active[lv], in->chunks_per_epoch, UINT64_MAX);
    CHECK_MUL_OVERFLOW(
      Error, per_lod_n_active[lv], in->covering_count, UINT64_MAX);
    const uint64_t batch_chunks =
      (uint64_t)per_lod_n_active[lv] * in->chunks_per_epoch;
    const uint64_t batch_covering =
      (uint64_t)per_lod_n_active[lv] * in->covering_count;
    CHECK_MUL_OVERFLOW(Error, batch_chunks, in->max_comp_chunk_bytes, SIZE_MAX);
    const size_t segment_bytes =
      (size_t)batch_chunks * in->max_comp_chunk_bytes;
    CHECK(Error, chunk_acc <= UINT64_MAX - batch_chunks);
    CHECK(Error, covering_acc <= UINT64_MAX - batch_covering);
    CHECK(Error, data_acc <= SIZE_MAX - segment_bytes);

    *seg = (struct lod_segment){
      .chunks_per_epoch = in->chunks_per_epoch,
      .covering_count = in->covering_count,
      .chunks_per_shard_inner = in->cps_inner,
      .n_active = per_lod_n_active[lv],
      .batch_chunk_offset = chunk_acc,
      .batch_covering_offset = covering_acc,
    };
    chunk_acc += batch_chunks;
    covering_acc += batch_covering;
    data_acc += segment_bytes;
  }

  out->total_batch_chunks = chunk_acc;
  out->total_batch_covering = covering_acc;
  out->total_data_bytes = data_acc;
  return 0;

Error:
  if (out)
    memset(out, 0, sizeof(*out));
  return 1;
}

int
aggregate_fixed_host_index(const struct batch_aggregate_layout* layout,
                           const struct aggregate_layout* per_lod,
                           size_t fixed_chunk_bytes,
                           size_t* offsets,
                           size_t* chunk_sizes)
{
  CHECK(Error, layout && per_lod && offsets && chunk_sizes);
  CHECK(Error, layout->total_batch_covering <= SIZE_MAX - layout->nlod);
  const size_t count = (size_t)layout->total_batch_covering + layout->nlod;
  CHECK(Error, count > 0);
  CHECK_MUL_OVERFLOW(Error, count, sizeof(*chunk_sizes), SIZE_MAX);
  memset(chunk_sizes, 0, count * sizeof(*chunk_sizes));

  for (uint8_t lv = 0; lv < layout->nlod; ++lv) {
    const struct lod_segment* seg = &layout->lods[lv];
    const struct aggregate_layout* agg = &per_lod[lv];
    CHECK(Error, seg->covering_count == agg->covering_count);
    CHECK(Error, seg->chunks_per_epoch == agg->chunks_per_epoch);
    CHECK(Error, seg->chunks_per_shard_inner == agg->cps_inner);
    CHECK(Error, agg->cps_inner > 0);
    CHECK(Error, agg->covering_count % agg->cps_inner == 0);
    const uint64_t num_shards = agg->covering_count / agg->cps_inner;
    const uint64_t shard_shape[2] = { num_shards, agg->cps_inner };
    const int64_t shard_strides[2] = {
      (int64_t)seg->n_active * (int64_t)agg->cps_inner,
      1,
    };
    const uint64_t metadata_base = seg->batch_covering_offset + lv;

    for (uint32_t a = 0; a < seg->n_active; ++a) {
      for (uint64_t j = 0; j < agg->chunks_per_epoch; ++j) {
        const uint64_t perm_pos =
          ravel(agg->lifted_rank, agg->lifted_shape, agg->lifted_strides, j);
        const uint64_t target = metadata_base +
                                ravel(2, shard_shape, shard_strides, perm_pos) +
                                (uint64_t)a * agg->cps_inner;
        CHECK(Error, target < count);
        chunk_sizes[target] = fixed_chunk_bytes;
      }
    }
  }

  size_t cursor = 0;
  for (size_t i = 0; i < count; ++i) {
    offsets[i] = cursor;
    CHECK(Error, chunk_sizes[i] <= SIZE_MAX - cursor);
    cursor += chunk_sizes[i];
  }
  CHECK(Error, cursor <= layout->total_data_bytes);
  return 0;

Error:
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
                         size_t page_size,
                         uint64_t chunks_per_shard_append)
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
  layout->page_size = page_size;
  layout->chunks_per_shard_append = chunks_per_shard_append;

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
