// Mirrors benchmarks/benchmark.py geometry in the parent acquire-zarr repo:
//   tyx = 1024 x 2048 x 2048 u16, 64-cube chunks, 1 x 16 x 16 chunks/shard
//   -> 64 shards of 128 MiB each.
// Used for thread-scaling and IO-impact studies against tensorstore.
//
// Layout is pinned: chunk_size and chunks_per_shard are set directly and
// chunk_ratios is left NULL so bench_util's resolve_chunk_sizing skips the
// planner (no dims_budget_chunk_bytes, no dims_set_shard_geometry). This
// preserves the exact (1, 16, 16) shard layout that benchmark.py uses; the
// planner's target_concurrent_shards/min_shard_bytes hints would otherwise
// collapse it to 4 large shards.
#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 1024, 2048, 2048 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  uint64_t chunk_sizes[] = { 64, 64, 64 };
  dims_set_chunk_sizes(dims, rank, chunk_sizes);

  // 1 t-chunk x 16 y-chunks x 16 x-chunks per shard.
  dims[0].chunks_per_shard = 1;
  dims[1].chunks_per_shard = 16;
  dims[2].chunks_per_shard = 16;

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = NULL,
                           });
}
