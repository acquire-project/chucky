#include "stream/types.aggregate.h"

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
