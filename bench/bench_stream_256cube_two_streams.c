#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[5];
  uint64_t sizes[] = { 1000, 256, 256, 256, 3 };
  uint8_t rank = dims_create(dims, "tzyxc", sizes);

  int ratios[] = { 1, 4, 4, 4, 0 };

  return bench_two_streams_main(ac,
                                av,
                                (struct bench_spec){
                                  .label = "two-streams",
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
