#pragma once

#include "stream.gpu.h"
#include <stddef.h>
#include <stdint.h>

struct mem_writer
{
  struct writer base;
  uint8_t* buf;
  size_t capacity;
  size_t cursor;
};

struct mem_writer
mem_writer_new(size_t capacity);

void
mem_writer_free(struct mem_writer* w);
