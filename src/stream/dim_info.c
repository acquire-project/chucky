#include "stream/dim_info.h"
#include "defs.limits.h"
#include "lod/lod_plan.h"

#include "util/prelude.h"

#include <assert.h>
#include <string.h>

uint64_t
dim_info_append_padding(const struct dim_info* info,
                        uint64_t cursor_elements,
                        int level)
{
  uint64_t appended[HALF_MAX_RANK];
  dim_info_final_append_sizes(info, cursor_elements, level, appended);
  const uint64_t chunk_size = info->append.beg[0].chunk_size;
  const uint64_t filled = appended[0] % chunk_size;
  return filled ? chunk_size - filled : 0;
}

void
dim_info_finalized_append_sizes(const struct dim_info* info,
                                uint64_t finalized_append_chunks,
                                uint64_t append_padding,
                                uint64_t* append_sizes)
{
  dim_info_decompose_append_sizes(info, finalized_append_chunks, append_sizes);

  // A flush that failed before delivering the chunk it padded records padding
  // no finalized chunk holds.
  const uint64_t reach = append_sizes[0];
  append_sizes[0] = append_padding < reach ? reach - append_padding : 0;
}

void
dim_info_decompose_append_sizes(const struct dim_info* info,
                                uint64_t total_append_chunks,
                                uint64_t* append_sizes)
{
  uint8_t na = dim_info_n_append(info);
  for (uint8_t i = 0; i < na; ++i)
    append_sizes[i] = 0;
  for (const struct dimension* d = info->append.beg + 1; d < info->append.end;
       ++d)
    append_sizes[dim_index(info, d)] = d->size;
  const uint64_t bac = info->bounded_append_chunks;
  assert(bac > 0 && "bounded_append_chunks must be > 0 after valid init");
  append_sizes[0] =
    ceildiv(total_append_chunks, bac) * info->append.beg[0].chunk_size;
}

int
dim_info_init(struct dim_info* info, const struct dimension* dims, uint8_t rank)
{
  CHECK(Fail, info);
  CHECK(Fail, dims_validate(dims, rank) == 0);

  memset(info, 0, sizeof(*info));

  uint8_t na = dims_n_append(dims, rank);

  info->append = (struct dim_slice){ .beg = dims, .end = dims + na };
  info->inner = (struct dim_slice){ .beg = dims + na, .end = dims + rank };

  // --- Precompute derived values ---

  // LOD mask: only inner dims with downsample
  info->lod_mask = 0;
  for (const struct dimension* d = info->inner.beg; d < info->inner.end; ++d) {
    if (d->downsample)
      info->lod_mask |= (1u << dim_index(info, d));
  }

  // Append downsample: rightmost append dim has downsample
  info->append_downsample = (na > 0) && dims[na - 1].downsample;

  // bounded_append_chunks: product of chunk_count for bounded append
  // dims 1..na-1
  info->bounded_append_chunks = 1;
  for (int d = 1; d < na; ++d)
    info->bounded_append_chunks *= ceildiv(dims[d].size, dims[d].chunk_size);

  return 0;
Fail:
  memset(info, 0, sizeof(*info));
  return 1;
}
