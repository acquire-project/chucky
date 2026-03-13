#include "bench_util.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  struct dimension dims[] = {
    { .size = 1000, .tile_size = 2, .tiles_per_shard = 32, .name = "t", .storage_position = 0 },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "z", .downsample = 1, .storage_position = 1 },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "y", .downsample = 1, .storage_position = 2 },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "x", .downsample = 1, .storage_position = 3 },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3, .name = "c", .storage_position = 4 },
  };
  return bench_stream_main(ac, av, "multiscale", dims, countof(dims));
}
