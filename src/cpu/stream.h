#pragma once

#include "stream.config.h"
#include "types.stream.h"

struct tile_stream_cpu;

// Create a CPU streaming pipeline. Returns NULL on failure or f16 dtype.
struct tile_stream_cpu*
tile_stream_cpu_create(const struct tile_stream_configuration* config);

void
tile_stream_cpu_destroy(struct tile_stream_cpu* s);

struct stream_metrics
tile_stream_cpu_get_metrics(const struct tile_stream_cpu* s);

const struct tile_stream_layout*
tile_stream_cpu_layout(const struct tile_stream_cpu* s);

struct writer*
tile_stream_cpu_writer(struct tile_stream_cpu* s);

uint64_t
tile_stream_cpu_cursor(const struct tile_stream_cpu* s);
