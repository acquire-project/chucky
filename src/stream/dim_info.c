#include "stream/dim_info.h"

#include "defs.limits.h"
#include "util/prelude.h"

#include <string.h>

int
dim_info_init(struct dim_info* info,
              const struct dimension* dims,
              uint8_t rank)
{
  CHECK(Fail, info);
  CHECK(Fail, dims);
  CHECK(Fail, rank > 0 && rank <= HALF_MAX_RANK);

  memset(info, 0, sizeof(*info));

  uint8_t na = dims_n_append(dims, rank);

  info->append = (struct dim_slice){ .beg = dims, .end = dims + na };
  info->inner = (struct dim_slice){ .beg = dims + na, .end = dims + rank };

  // --- Validate constraints ---

  // chunk_size > 0
  for (int d = 0; d < rank; ++d) {
    if (dims[d].chunk_size == 0) {
      log_error("dims[%d].chunk_size must be > 0", d);
      goto Fail;
    }
  }

  // Only dim 0 may be unbounded
  for (int d = 1; d < rank; ++d) {
    if (dims[d].size == 0) {
      log_error("dims[%d].size must be > 0 (only dim 0 may be unbounded)", d);
      goto Fail;
    }
  }

  // Unbounded dim 0 requires chunks_per_shard > 0
  if (dims[0].size == 0 && dims[0].chunks_per_shard == 0) {
    log_error("dims[0].size=0 (unbounded) requires chunks_per_shard > 0");
    goto Fail;
  }

  // Append dims pinned to identity storage positions
  for (int d = 0; d < na; ++d) {
    if (dims[d].storage_position != d) {
      log_error("dims[%d].storage_position must be %d (append dim)", d, d);
      goto Fail;
    }
  }

  // Valid permutation
  {
    uint32_t seen = 0;
    for (int d = 0; d < rank; ++d) {
      uint8_t j = dims[d].storage_position;
      if (j >= rank || (seen & (1u << j))) {
        log_error("invalid storage_position permutation at dims[%d]", d);
        goto Fail;
      }
      seen |= (1u << j);
    }
  }

  // --- Precompute derived values ---

  // LOD mask: only inner dims with downsample
  info->lod_mask = 0;
  for (const struct dimension* d = info->inner.beg; d < info->inner.end; ++d) {
    if (d->downsample)
      info->lod_mask |= (1u << dim_index(info, d));
  }

  // Append downsample: rightmost append dim has downsample
  info->append_downsample = (na > 0) && dims[na - 1].downsample;

  // inner_append_count: product of chunk_count for bounded append dims 1..na-1
  info->inner_append_count = 1;
  for (int d = 1; d < na; ++d)
    info->inner_append_count *= ceildiv(dims[d].size, dims[d].chunk_size);

  return 0;
Fail:
  memset(info, 0, sizeof(*info));
  return 1;
}
