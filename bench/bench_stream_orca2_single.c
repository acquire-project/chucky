#include "bench_util.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  struct dimension dims[] = {
    { .size = 10000, .tile_size = 64, .tiles_per_shard = 10, .name = "t" },
    { .size = 2, .tile_size = 1, .tiles_per_shard = 2, .name = "c" },
    { .size = 2048, .tile_size = 64, .tiles_per_shard = 10, .name = "y" },
    { .size = 2304, .tile_size = 64, .tiles_per_shard = 10, .name = "x" },
  };
  return bench_stream_main(ac, av, "single", dims, countof(dims));
}
