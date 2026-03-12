#include "bench_util.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  struct dimension dims[] = {
    {
      .size = 100000,
      .tile_size = 64,
      .tiles_per_shard = 20,
      .name = "t",
      .downsample = 1,
    },
    {
      .size = 2,
      .tile_size = 1,
      .tiles_per_shard = 2,
      .name = "c",
    },
    {
      .size = 2048,
      .tile_size = 64,
      .tiles_per_shard = 10,
      .name = "y",
      .downsample = 1,
    },
    {
      .size = 2304,
      .tile_size = 64,
      .tiles_per_shard = 10,
      .name = "x",
      .downsample = 1,
    },
  };
  return bench_stream_main(ac, av, "multiscale_dim0", dims, countof(dims));
}
