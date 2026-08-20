// smallepoch_single over four shard files instead of one. Its twin pins y and
// x at their full extent, which leaves one chunk along each and so exactly one
// shard file open at a time — no two writes can ever be outstanding. Splitting
// y and x into two chunks each gives four. The chunk stays 1 MiB, so the epoch
// here is four chunks and four times the twin's; the two scenarios are worth
// comparing on what the write path did, not on throughput. Keep both: the
// one-file case is the only place pre-sizing a file can be measured.
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
