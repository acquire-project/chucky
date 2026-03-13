#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 10000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  uint8_t ratios[] = { 6, 0, 6, 6 };
  dims_budget_tile_size(dims, rank, 1ULL << 18, ratios);

  uint64_t shard_counts[] = { 16, 1, 4, 4 };
  dims_set_shard_counts(dims, rank, shard_counts);

  return bench_stream_main(ac, av, "single", dims, rank);
}
