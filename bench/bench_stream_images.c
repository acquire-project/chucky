#include "bench_util.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  const uint64_t sizes[] = { 4, 64, 64 };
  uint8_t rank = dims_create(dims, "tyx", sizes);
  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "images",
                             .dims = dims,
                             .rank = rank,
                             .image_input = 1,
                             .min_shard_bytes = 512ull << 20,
                             .max_shard_bytes = 1ull << 30,
                             .target_concurrent_shards = 4,
                             .min_append_shards = 0,
                           });
}
