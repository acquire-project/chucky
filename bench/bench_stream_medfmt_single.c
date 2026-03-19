#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 10, 1, 10240, 15360 };
  uint8_t rank = dims_create(dims, "zcyx", sizes);

  dims_set_downsample_by_name(dims, rank, "zyx");

  uint8_t ratios[] = { 1, 0, 4, 4 };
  uint64_t shard_counts[] = { 1, 1, 4, 4 };

  return bench_stream_main(
    ac, av, "single", dims, rank, ratios, 1 << 18, shard_counts);
}
