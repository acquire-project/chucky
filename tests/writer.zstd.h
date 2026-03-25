#pragma once

#include "stream.gpu.h"
#include <stddef.h>
#include <stdint.h>

struct zstd_chunk_writer
{
  struct tile_writer base;
  struct writer* sink; // not owned
  size_t chunk_bytes;  // expected decompressed size
  uint8_t* decomp_buf; // scratch buffer (one chunk)
  size_t total_compressed;
  size_t total_decompressed;
};

struct zstd_chunk_writer
zstd_chunk_writer_new(size_t chunk_bytes, struct writer* sink);

void
zstd_chunk_writer_free(struct zstd_chunk_writer* w);
