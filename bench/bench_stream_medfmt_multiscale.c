#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 100, 1, 10240, 15360 };
  uint8_t rank = dims_create(dims, "zcyx", sizes);

  dims_set_downsample_by_name(dims, rank, "yx");

  int ratios[] = { 1, 0, 8, 8 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "multiscale",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 18,
                             .min_chunk_bytes = 1 << 12,
                             .min_shard_bytes = 1ull << 30,
                             .target_concurrent_shards = 20,
                             .min_append_shards = 4,
                           });
}
