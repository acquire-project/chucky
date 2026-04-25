// Mirrors benchmarks/benchmark.py geometry in the parent acquire-zarr repo:
//   tyx = 1024 x 2048 x 2048 u16, 64-cube chunks, 16x16 xy shards.
// Used for thread-scaling and IO-impact studies against tensorstore.
#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 1024, 2048, 2048 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  // Equal weights -> 64-cube chunks at target_chunk_bytes = 64^3 * 2 = 512 KiB.
  int ratios[] = { 1, 1, 1 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 19, // 512 KiB
                             .min_chunk_bytes = 1 << 14,
                             // 1 x 16 x 16 chunks/shard -> 128 MiB shards.
                             .min_shard_bytes = 128ull << 20,
                             .target_concurrent_shards = 4,
                             .min_append_shards = 1,
                           });
}
