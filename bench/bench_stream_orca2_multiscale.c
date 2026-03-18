#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  dims_set_downsample_by_name(dims, rank, "yx");

  uint8_t ratios[] = { 1, 0, 2, 2 };
  uint64_t shard_counts[] = { 2, 1, 2, 2 };

  return bench_stream_main(ac, av, "multiscale", dims, rank, ratios, 1 << 17,
                           shard_counts);
}
