#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  int ratios[] = { 1, 0, 2, 2 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 20,
                             .min_chunk_bytes = 1 << 14,
                             .min_shard_bytes = 1ull << 30,
                             .target_concurrent_shards = 16,
                             .min_append_shards = 4,
                           });
}
