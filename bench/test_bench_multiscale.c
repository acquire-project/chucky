#include "bench_util.h"

#include <string.h>

int
main(int argc, char** argv)
{
  if (argc < 2)
    return 1;
  struct dimension dims[3];
  const uint8_t rank = dims_create(dims, "tyx", (uint64_t[]){ 1024, 32, 32 });
  dims_set_downsample_by_name(
    dims, rank, strcmp(argv[1], "append") == 0 ? "tyx" : "yx");
  const int ratios[] = { 0, 1, 1 };
  return bench_stream_main(argc - 1,
                           argv + 1,
                           (struct bench_spec){
                             .label = "multiscale",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 128,
                             .min_shard_bytes = 1 << 18,
                             .target_concurrent_shards = 1,
                             .min_append_shards = 4,
                           });
}
