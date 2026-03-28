#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[5];
  uint64_t sizes[] = { 1000, 256, 256, 256, 3 };
  uint8_t rank = dims_create(dims, "tzyxc", sizes);

  uint8_t ratios[] = { 1, 4, 4, 4, 0 };
  uint64_t shard_counts[] = { 16, 2, 2, 2, 1 };

  return bench_two_streams_main(
    ac, av, "two-streams", dims, rank, ratios, 1 << 18, shard_counts);
}
