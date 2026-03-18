#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[5];
  uint64_t sizes[] = { 1000, 256, 256, 256, 3 };
  uint8_t rank = dims_create(dims, "tzyxc", sizes);

  dims_set_storage_order(dims, rank, "tczyx");
  dims_set_downsample_by_name(dims, rank, "tzyx");

  uint8_t ratios[] = { 0, 1, 1, 1, 0 };
  uint64_t shard_counts[] = { 16, 2, 2, 2, 1 };

  return bench_stream_main(ac, av, "multiscale_dim0", dims, rank, ratios,
                           1 << 17, shard_counts);
}
