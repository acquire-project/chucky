#pragma once

#include "bench_parse.h"
#include "bench_report.h"
#include "bench_zarr.h"
#include "dimension.h"
#include "dtype.h"
#include "sink_discard.h"
#include "sink_metering.h"
#include "sink_throttled.h"
#include "stream.gpu.h"
#include "test_data.h"
#include "types.codec.h"
#include "types.lod.h"
#include "writer.h"

#include <stddef.h>
#include <stdint.h>

struct bench_config
{
  const char* label;
  struct dimension* dims;
  uint8_t rank;
  fill_fn fill;
  const char* output_path;
  const char* array_name;
  const char* s3_bucket;     // NULL = no S3 output
  const char* s3_prefix;     // key prefix (no leading/trailing /)
  const char* s3_region;     // e.g. "us-east-1"
  const char* s3_endpoint;   // e.g. "https://s3.us-east-1.amazonaws.com"
  double s3_throughput_gbps; // gigabits/s, 0 = CRT default (10.0)
  struct codec_config codec;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method append_reduce_method;
  enum bench_backend backend;
  enum dtype dtype;          // element type (default dtype_u16)
  const int* chunk_ratios;   // power-of-2 distribution ratios; see
                             // dims_budget_chunk_size for the -1/0/>0
                             // conventions
  size_t target_chunk_bytes; // 0 = use 1MB default
  size_t min_chunk_bytes;    // auto-fit floor; 0 = no floor
  size_t target_batch_bytes; // 0 = 512 MiB default; controls auto-K
  size_t memory_budget;      // 0 = auto-detect
  size_t min_shard_bytes;    // minimum uncompressed bytes per shard
  uint32_t
    target_concurrent_shards; // cap on inner shard product (active files)
  uint32_t min_append_shards; // require at least N shards along the outer
                              // append dim (0 = no minimum). Forces
                              // shard-switching in benchmarks that would
                              // otherwise collapse to a single shard.
  int json_output;            // print JSON to stdout after run
  uint64_t io_bw_mbps;        // 0 = no bandwidth cap (MiB/s)
  uint64_t io_latency_us;     // 0 = no fixed per-job latency
  size_t backpressure_bytes;  // 0 = disabled; >0 = stall when pending > N
  int max_threads;            // 0 = OpenMP default (omp_get_max_threads)
  int prefill;                // fill the pump buffer once, append repeatedly
};

int
run_bench(const struct bench_config* cfg);

// Bench-author-facing spec: everything a bench main() picks before CLI parsing.
struct bench_spec
{
  const char* label;
  struct dimension* dims;
  uint8_t rank;
  const int* chunk_ratios;
  size_t target_chunk_bytes;
  size_t min_chunk_bytes; // auto-fit floor; bench fails if budget
                          // can't meet it (0 = no floor)
  size_t min_shard_bytes; // minimum uncompressed bytes per shard
  uint32_t
    target_concurrent_shards; // cap on inner shard product (active files)
  uint32_t min_append_shards; // 0 = no minimum (see bench_config)
};

// CLI driver: parses --fill, --codec, --reduce, --dtype, --frames, --json,
// -o flags, inits CUDA, calls run_bench, handles xor_pattern_init/free.
int
bench_stream_main(int ac, char* av[], struct bench_spec spec);

// Two-stream variant: creates two GPU pipelines on the same CUDA context and
// interleaves writer_append calls for balanced GPU utilisation.
int
bench_two_streams_main(int ac, char* av[], struct bench_spec spec);
