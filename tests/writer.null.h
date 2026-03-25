#pragma once

#include "stream.gpu.h"
#include <stddef.h>

struct null_writer
{
  struct writer base;
  struct tile_writer chunk_base;
  size_t bytes_discarded;
  size_t chunks_discarded;
  size_t compressed_bytes;
};

struct null_writer
null_writer_new(void);
