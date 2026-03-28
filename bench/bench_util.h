#include "stream/layouts.h"
#pragma once

#include <stddef.h>

#include "gpu/compress.h"
#include "gpu/lod.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "test_data.h"
#include "util/metric.h"

#include <stddef.h>
#include <stdint.h>

double
gb_per_s(double bytes, double ms);

#define print_report(...) fprintf(stderr, __VA_ARGS__), fprintf(stderr, "\n")

// --- Discard shard_sink for benchmarks ---

struct discard_shard_writer
{
  struct shard_writer base;
  struct discard_shard_sink* parent;
};

struct discard_shard_sink
{
  struct shard_sink base;
  struct discard_shard_writer writer;
  size_t total_bytes;
  size_t shards_finalized;
};

void
discard_shard_sink_init(struct discard_shard_sink* s);

// --- Metering shard_sink wrapper ---

#define METER_MAX_WRITERS 32

struct metering_writer
{
  struct shard_writer base;
  struct shard_writer* inner;
  struct metering_sink* parent;
  int in_use;
};

struct metering_sink
{
  struct shard_sink base;
  struct shard_sink* inner;
  struct metering_writer writers[METER_MAX_WRITERS];
  size_t total_bytes;
  struct stream_metric metric;
  struct platform_clock clock;
};

void
metering_sink_init(struct metering_sink* ms, struct shard_sink* inner);

// --- Report + pipeline helpers ---

struct sink_stats
{
  size_t total_bytes;
  uint64_t total_chunks; // all LOD levels, per epoch
};

void
print_metric_row(const struct stream_metric* m);
void
log_bench_header(const struct tile_stream_layout* layout,
                 enum dtype dtype,
                 enum compression_codec codec,
                 size_t max_compressed_size,
                 size_t codec_batch_size,
                 size_t total_bytes,
                 size_t total_elements);
void
print_bench_report(const struct stream_metrics* metrics,
                   const struct tile_stream_layout* layout,
                   enum dtype dtype,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s,
                   float init_s,
                   float flush_s,
                   size_t flush_pending_bytes);

enum bench_backend
{
  BENCH_GPU,
  BENCH_CPU,
};

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
  enum compression_codec codec;
  enum lod_reduce_method reduce_method;
  enum lod_reduce_method append_reduce_method;
  enum bench_backend backend;
  enum dtype dtype;             // element type (default dtype_u16)
  const uint8_t* chunk_ratios;  // power-of-2 distribution ratios
  size_t target_chunk_bytes;    // 0 = use 1MB default
  size_t memory_budget;         // 0 = auto-detect
  const uint64_t* shard_counts; // per-dim target shard counts (NULL = skip)
  int json_output;              // print JSON to stdout after run
};

int
run_bench(const struct bench_config* cfg);

// CLI driver: parses --fill, --codec, --reduce, --dtype, --frames, --json,
// -o flags, inits CUDA, calls run_bench, handles xor_pattern_init/free.
int
bench_stream_main(int ac,
                  char* av[],
                  const char* label,
                  struct dimension* dims,
                  uint8_t rank,
                  const uint8_t* chunk_ratios,
                  size_t default_chunk_bytes,
                  const uint64_t* shard_counts);

// Two-stream variant: creates two GPU pipelines on the same CUDA context and
// interleaves writer_append calls for balanced GPU utilisation.
int
bench_two_streams_main(int ac,
                       char* av[],
                       const char* label,
                       struct dimension* dims,
                       uint8_t rank,
                       const uint8_t* chunk_ratios,
                       size_t default_chunk_bytes,
                       const uint64_t* shard_counts);
