#pragma once

#include "bench_input.h"
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

struct bench_append_sample
{
  uint64_t calls;
  uint64_t over_100ms;
  double total_ms;
  double max_ms;
};

struct bench_measurement
{
  int boundary_timing;
  int64_t start_ns; // private clock origin, omitted from reports
  double prep_s, warmup_s, warmup_drain_s, append_s, elapsed_s, drain_s;
  double requested_warmup_s, requested_duration_s;
  double target_duration_s, discarded_attempts_s;
  unsigned attempt, max_attempts;
  uint64_t requested_frames, warmup_bytes, warmup_output_bytes;
  uint64_t epoch_bytes, epochs_per_batch, staging_bytes;
  uint64_t memory_budget, target_batch_bytes;
  uint64_t complete_batches, batch_reuses, generation_transitions;
  int coverage_sufficient;
  uint8_t rank;
  struct dimension geometry[HALF_MAX_RANK];
  uint64_t reference_frames, source_bytes, append_bytes;
  uint64_t input_bytes, output_bytes;
  uint64_t logical_input_bytes;
  uint64_t boundary_bytes[3];
  struct bench_append_sample boundary[3], following[3];
};

struct bench_image_report
{
  const struct bench_input* input;
  const char* backend;
  float context_init_s;
};

void
print_measurement_report(const struct bench_measurement* run);

void
print_memory_report(const struct bench_memory* mem);

void
print_stage_report(const struct stream_metrics* m);

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
                      int worker_threads,
                      const struct bench_measurement* measurement,
                      const struct bench_image_report* images);

// Emit a minimal error JSON (`{"status":"error"}`) to stdout.
void
print_bench_json_error(void);

void
print_bench_json_coverage_error(const struct bench_measurement* measurement);
