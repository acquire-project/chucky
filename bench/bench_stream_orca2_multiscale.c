#include "bench_util.h"
#include "log/log.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  const uint64_t t_nelem = 1 << 16;
  const uint64_t t_xy = 64;
  const uint64_t s_xy = 2;
  const uint64_t s_z = 2;

  const uint64_t tps_xy = (2304 + t_xy * s_xy - 1) / (t_xy * s_xy);
  const uint64_t t_z = t_nelem / t_xy / t_xy;
  const uint64_t tps_z = (1000 + t_z * s_z - 1) / (t_z * s_z);

  log_info("t_xy: %d\tt_z: %d", (int)t_xy, (int)t_z);

  struct dimension dims[] = {
    {
      .size = 1000,
      .tile_size = t_z,
      .tiles_per_shard = tps_z,
      .name = "t",
    },
    {
      .size = 2,
      .tile_size = 1,
      .tiles_per_shard = 2,
      .name = "c",
    },
    {
      .size = 2048,
      .tile_size = t_xy,
      .tiles_per_shard = tps_xy,
      .name = "y",
      .downsample = 1,
    },
    {
      .size = 2304,
      .tile_size = t_xy,
      .tiles_per_shard = tps_xy,
      .name = "x",
      .downsample = 1,
    },
  };
  return bench_stream_main(ac, av, "multiscale", dims, countof(dims));
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
