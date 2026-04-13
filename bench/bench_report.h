#pragma once

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
print_metric_row(const struct stream_metric* m);

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
                   size_t flush_pending_bytes);
