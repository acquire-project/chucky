#pragma once

#include "bench_memory.h"
#include "stream/layouts.h"
#include "types.stream.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define print_report(...) fprintf(stderr, __VA_ARGS__), fprintf(stderr, "\n")

double
gb_per_s(double bytes, double ms);

struct sink_stats
{
  size_t total_bytes;
  uint64_t total_chunks; // all LOD levels, per epoch
};

void
print_memory_report(const struct bench_memory* mem);

void
print_metric_row(const struct stream_metric* m);

// Print diagnostic intervals grouped by where the work or wait happened.
// Unlike stage rows, these intervals do not claim a byte rate.
void
print_diagnostics_report(const struct stream_metrics* metrics, float wall_s);

// The time taken per append is printed here.
void
print_append_latency(const struct stream_metrics* m);

void
log_bench_header(const struct tile_stream_layout* layout,
                 enum dtype dtype,
                 struct codec_config codec,
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
                   uint64_t flush_pending_bytes);

// Emit the pass-case JSON report to stdout. sink_metric may be NULL (no sink
// block is written in that case).
void
print_bench_json_pass(const struct stream_metrics* metrics,
                      const struct stream_metric* sink_metric,
                      const struct tile_stream_layout* layout,
                      enum dtype dtype,
                      struct codec_config codec,
                      const struct sink_stats* ss,
                      size_t total_bytes,
                      size_t total_elements,
                      float wall_s,
                      float init_s,
                      float flush_s,
                      const struct bench_memory* mem,
                      int worker_threads);

// Emit a minimal error JSON (`{"status":"error"}`) to stdout.
void
print_bench_json_error(void);
