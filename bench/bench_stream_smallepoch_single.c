#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 65536, 16, 16 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  uint8_t ratios[] = { 1, 0, 0 };
  uint64_t shard_counts[] = { 64, 1, 1 };

  return bench_stream_main(
    ac, av, "single", dims, rank, ratios, 1 << 16, shard_counts);
}
