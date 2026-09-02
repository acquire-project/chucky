#pragma once

#include "stream/layouts.h"
#include "types.stream.h"
#include "zarr/types.io.h"

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

// Both the estimated and the measured memory use are recorded here.
struct bench_memory
{
  uint64_t estimate_total_bytes;  // device bytes on GPU, heap bytes on CPU
  uint64_t estimate_pinned_bytes; // pinned host bytes on GPU, 0 on CPU
  uint64_t host_baseline_bytes;   // resident before the stream was created
  uint64_t host_peak_bytes;       // most resident memory held during the run
  // A reading of 0 is valid, so it cannot signal failure.
  int host_reading_failed;
  uint64_t device_used_bytes; // GPU: free device memory the stream took
  // Device memory on GPU, the host difference on CPU. 0 if unavailable.
  uint64_t measured_bytes;
};

// The write scheduling a run used, for the results file. Recorded even when
// nothing was written, so a run with no output is still told apart from one
// taken with different settings.
struct io_write_scheduling
{
  struct io_scheduling io;
  const char* backend; // NULL when the run wrote nothing
  uint64_t host_output_budget_bytes;
};

void
print_memory_report(const struct bench_memory* mem);

void
print_metric_row(const struct stream_metric* m);

// Print diagnostic intervals in separate wait, pipeline-gap, and host-work
// sections. Unlike stage rows, these intervals do not claim a byte rate.
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
                   uint64_t flush_pending_bytes,
                   const struct shard_pool_io_stats* io);

// Emit the pass-case JSON report to stdout. sink_metric may be NULL (no sink
// block is written in that case).
void
print_bench_json_pass(const struct stream_metrics* metrics,
                      const struct stream_metric* sink_metric,
                      const struct tile_stream_layout* layout,
                      enum dtype dtype,
                      const struct sink_stats* ss,
                      size_t total_bytes,
                      size_t total_elements,
                      float wall_s,
                      float init_s,
                      float flush_s,
                      const struct bench_memory* mem,
                      int worker_threads,
                      const struct io_write_scheduling* scheduling,
                      const struct shard_pool_io_stats* io);

// Emit a minimal error JSON (`{"status":"error"}`) to stdout.
void
print_bench_json_error(void);
