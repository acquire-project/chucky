#pragma once

#include "gpu/stream.engine.h"

struct tile_stream_gpu
{
  struct writer writer;
  struct stream_engine engine;
  struct stream_context ctx;
  // Owner of the per-array allocations; the engine holds the bound copy.
  struct engine_array_state ar;
  int flushed; // 1 after flush; reset by append for idempotency
};

// Set writer vtable (append/flush).
void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s);
