#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 10000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  dims_set_downsample_by_name(dims, rank, "tyx");

  uint8_t ratios[] = { 6, 0, 6, 6 };
  uint64_t shard_counts[] = { 8, 1, 4, 4 };

  return bench_stream_main(ac, av, "multiscale_dim0", dims, rank, ratios,
                           1 << 19, shard_counts);
}
