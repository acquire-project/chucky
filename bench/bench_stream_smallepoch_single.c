#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 1 << 24, 16, 16 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  int ratios[] = { 1, -1, -1 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 16,
                             .min_chunk_bytes = 1 << 12,
                             .min_shard_bytes = 1 << 16,
                             .target_concurrent_shards = 1,
                             .min_append_shards = 4,
                           });
}
