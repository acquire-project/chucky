#pragma once

#include "multiarray/multiarray.h"
#include "types.stream.h"

struct multiarray_tile_stream_cpu;
struct stream_metrics;

// Create a multiarray stream.  Pass enable_metrics != 0 to collect timing.
struct multiarray_tile_stream_cpu*
multiarray_tile_stream_cpu_create(
  int n_arrays,
  const struct tile_stream_configuration configs[],
  struct shard_sink* sinks[],
  int enable_metrics);

void
multiarray_tile_stream_cpu_destroy(struct multiarray_tile_stream_cpu* ms);

struct multiarray_writer*
multiarray_tile_stream_cpu_writer(struct multiarray_tile_stream_cpu* ms);

struct stream_metrics
multiarray_tile_stream_cpu_get_metrics(
  const struct multiarray_tile_stream_cpu* ms);
