// Four shard files where its twin holds one. The twin pins y and x at full
// extent, so no two of its writes can ever be outstanding; splitting each in
// two gives four. The chunk stays 1 MiB, so this epoch is four times the
// twin's. Compare the pair on what the write path did, not on throughput.
// The twin stays: one growing file is the only place pre-sizing can be
// measured.
#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 1 << 20, 16, 16 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  // 19 bits of a 1 MiB chunk at 2 bytes per element, split 13/3/3: t 8192,
  // y 8, x 8. That leaves 2 chunks along y and 2 along x, which is the four.
  int ratios[] = { 13, 3, 3 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .target_chunk_bytes = 1 << 20,
                             .min_chunk_bytes = 1 << 12,
                             .min_shard_bytes = 1 << 20,
                             .target_concurrent_shards = 4,
                             .min_append_shards = 4,
                           });
}
