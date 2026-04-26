#pragma once

#include "writer.h"

#include <stdatomic.h>
#include <stdint.h>

#define COUNTING_SINK_MAX_WRITERS 32

struct counting_writer
{
  struct shard_writer base;
  struct shard_writer* inner;
  struct counting_sink* parent;
  int in_use;
};

// Test-only shard_sink shim. Wraps an inner sink and counts how many calls
// went through the copy path (write) vs the zero-copy path (write_direct).
// All other vtable methods forward to the inner sink unchanged, so the
// stream sees the same alignment requirement and async-IO semantics.
struct counting_sink
{
  struct shard_sink base;
  struct shard_sink* inner;
  struct counting_writer writers[COUNTING_SINK_MAX_WRITERS];
  _Atomic uint64_t copy_calls;
  _Atomic uint64_t direct_calls;
  _Atomic uint64_t copy_bytes;
  _Atomic uint64_t direct_bytes;
};

void
counting_sink_init(struct counting_sink* cs, struct shard_sink* inner);

void
counting_sink_path_counts(const struct counting_sink* cs,
                          uint64_t* out_copy_calls,
                          uint64_t* out_direct_calls,
                          uint64_t* out_copy_bytes,
                          uint64_t* out_direct_bytes);
