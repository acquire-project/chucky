#include "stream/types.aggregate.h"

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
      out_gather[idx] =
        (uint32_t)(pool_epoch * total_chunks + levels->chunk_offset[lv] + j);

      uint64_t perm_pos =
        ravel(agg->lifted_rank, agg->lifted_shape, agg->lifted_strides, j);
      out_perm[idx] =
        (uint32_t)(ravel(2, shard_shape, shard_strides, perm_pos) +
                   a * cps_inner);
    }
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
  layout->page_size = page_size;

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
