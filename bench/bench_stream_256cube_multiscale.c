#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[5];
  uint64_t sizes[] = { 1000, 256, 256, 256, 3 };
  uint8_t rank = dims_create(dims, "tzyxc", sizes);

  uint8_t ratios[] = { 1, 4, 4, 4, 0 };
  dims_budget_tile_size(dims, rank, 1ULL << 13, ratios);

  uint64_t shard_counts[] = { 16, 2, 2, 2, 1 };
  dims_set_shard_counts(dims, rank, shard_counts);

  dims_set_downsample_by_name(dims, rank, "zyx");

  return bench_stream_main(ac, av, "multiscale", dims, rank);
}
