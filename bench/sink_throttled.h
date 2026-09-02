#pragma once

#include "writer.h"
#include "zarr/io_scheduler.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

struct throttled_shard_writer
{
  struct shard_writer base;
  struct throttled_shard_sink* parent;
};

struct throttled_shard_sink
{
  struct shard_sink base;
  struct throttled_shard_writer writer; // single shared writer
  struct io_scheduler* scheduler;       // owned
  _Atomic uint64_t total_bytes;         // reporting
  uint64_t latency_ns;                  // fixed per-job cost
  uint64_t bytes_per_sec;               // 0 = no bandwidth cap
};

// Initialize a throttled sink. Returns 0 on success.
int
throttled_shard_sink_init(struct throttled_shard_sink* s,
                          uint64_t io_bw_mbps,
                          uint64_t io_latency_us);

// Drain the queue and free resources.
void
throttled_shard_sink_teardown(struct throttled_shard_sink* s);
