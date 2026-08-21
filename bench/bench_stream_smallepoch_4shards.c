// Four shard files where the twin has one; compare writes, not throughput.
#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 1 << 20, 16, 16 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  // 13/3/3 of 19 bits: t 8192, y 8, x 8 - 2 chunks each along y and x.
  int ratios[] = { 13, 3, 3 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 20,
                             .min_chunk_bytes = 1 << 12,
                             .min_shard_bytes = 1 << 20,
                             .target_concurrent_shards = 4,
                             .min_append_shards = 4,
                           });
}
