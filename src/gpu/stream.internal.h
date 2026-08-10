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
  // Captured at create. The streams and device memory belong to it, and a
  // caller is free to append or flush from a thread that has no context of
  // its own, so every entry point makes it current first.
  CUcontext cuda;
};

// Make the stream's context current on the calling thread. Kernel launches go
// through the runtime API, which binds this thread's own context rather than
// the one that owns the stream, and rejects the launch when they differ.
void
tile_stream_gpu_bind_context(struct tile_stream_gpu* s);

// Set writer vtable (append/flush).
void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s);
