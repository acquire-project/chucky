#pragma once

#include "writer.h"

#include <stddef.h>
#include <stdint.h>

struct bench_input
{
  uint16_t* data;
  size_t elements;
  size_t frame_elements;
  size_t logical_frame_elements;
  size_t source_bytes;
  float load_s;
};

// Prepared cyclic input. All offsets and sizes are writer bytes; logical
// frame bytes exclude image edge padding. Preparation precedes measurement.
struct bench_source
{
  const unsigned char* data;
  size_t bytes;
  size_t frame_bytes;
  size_t logical_frame_bytes;
};

struct slice
bench_source_slice(const struct bench_source* source,
                   uint64_t offset,
                   size_t maximum_bytes);

int
bench_input_load(struct bench_input* input,
                 const char* path,
                 size_t width,
                 size_t height,
                 size_t chunk_width,
                 size_t chunk_height);

void
bench_input_free(struct bench_input* input);
