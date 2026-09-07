#pragma once

#include "writer.h"

#include <stddef.h>
#include <stdint.h>

struct bench_input
{
  uint16_t* data;
  size_t elements;
  size_t frame_elements;
  size_t source_bytes;
  float load_s;
};

int
bench_input_load(struct bench_input* input,
                 const char* path,
                 size_t width,
                 size_t height);

int
bench_input_pump(const struct bench_input* input,
                 struct writer* writer,
                 size_t total_elements,
                 size_t append_elements);

void
bench_input_free(struct bench_input* input);
