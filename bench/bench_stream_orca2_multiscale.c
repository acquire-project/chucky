#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  uint8_t ratios[] = { 1, 0, 2, 2 };
  dims_budget_tile_size(dims, rank, 1ULL << 16, ratios);

  uint64_t shard_counts[] = { 2, 1, 2, 2 };
  dims_set_shard_counts(dims, rank, shard_counts);

  dims_set_downsample_by_name(dims, rank, "yx");

  return bench_stream_main(ac, av, "multiscale", dims, rank);
}

/* NOTES
on auk, 1 MB/tile will oom. Epoch too thick.
Can fix by going to 0.5 MB/tile.

z  xy   t_nelem
16 128  1<<18     1.28 GB/s
4  256  1<<18     1.18 GB/s
64 64   1<<18     oom
32 64   1<<17     oom
16 64   1<<16     1.75 GB/s
4  128  1<<16     1.22 GB/s

adjusted min tiles (2048) and max epochs/batch (256)
4  256  1<<18     oom
2  256  1<<17     1.21 GB/s
8  128  1<<17     1.26 GB/s
14 96   1<<17     1.35 GB/s
16 64   1<<16     1.74 GB/s
 */
