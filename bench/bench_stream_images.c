#include "bench_util.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  const uint64_t sizes[] = { 1, 256, 256 };
  uint8_t rank = dims_create(dims, "tyx", sizes);
  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "images",
                             .dims = dims,
                             .rank = rank,
                             .image_input = 1,
                             .min_shard_bytes = 64 << 20,
                             .target_concurrent_shards = 16,
                             .min_append_shards = 4,
                           });
}
