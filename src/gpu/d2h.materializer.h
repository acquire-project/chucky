#pragma once

#include "gpu/aggregate.h"
#include "gpu/pool.h"
#include "stream/types.aggregate.h"
#include "zarr/host_batch.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct aggregate_layout;
struct gpu_ordering;
struct shard_state;

enum device_aggregate_extent_kind
{
  DEVICE_AGGREGATE_FIXED_EXTENT = 0,
  DEVICE_AGGREGATE_INDEXED_EXTENT,
};

// Immutable description handed from aggregation to materialization.  begin()
// turns the pool handles into the slot lease stored in its ticket.
struct device_aggregate_batch
{
  int slot_index;
  enum device_aggregate_extent_kind extent_kind;
  struct batch_aggregate_layout layout;
  uint8_t nlod;
  uint32_t per_lod_n_active[LOD_MAX_LEVELS];

  struct gpu_pool* aggregate_pool;
  struct gpu_pool* host_pool;
  struct gpu_pool* index_pool;
  CUevent completion;
};

// Placement becomes authoritative only at finish time.  PR 1's legacy
// materializers use it to construct run views over the existing device
// layout; PR 2 can use the same boundary to place committed host tails.
struct d2h_host_placement
{
  const struct aggregate_layout* per_lod_layouts;
  struct shard_state* const* shards_by_lod;
  void* slot_lifetime;
};

enum d2h_ticket_state
{
  D2H_TICKET_EMPTY = 0,
  D2H_TICKET_METADATA_PENDING,
  D2H_TICKET_PAYLOAD_PENDING,
  D2H_TICKET_HOST_READY,
  D2H_TICKET_ERROR,
};

struct d2h_ticket
{
  struct device_aggregate_batch batch;
  struct aggregate_slot* slot;
  struct d2h_transfer_span* spans;
  size_t span_count;
  size_t span_capacity;
  struct host_batch host;
  CUevent payload_start;
  enum d2h_ticket_state state;
};

struct d2h_materializer_ops;

struct d2h_materializer
{
  const struct d2h_materializer_ops* ops;
  enum device_aggregate_extent_kind extent_kind;
  struct gpu_ordering* ord;
  CUstream payload_stream;
  CUevent begin_event[2];
  CUevent payload_event[2];
  struct d2h_ticket ticket[2];
};

int
d2h_materializer_init(struct d2h_materializer* materializer,
                      enum device_aggregate_extent_kind extent_kind,
                      struct gpu_ordering* ord,
                      CUstream payload_stream,
                      CUstream seed_stream);

void
d2h_materializer_destroy(struct d2h_materializer* materializer);

// Starts every operation that is independent of committed prior-tail state.
int
d2h_materialize_begin(struct d2h_materializer* materializer,
                      const struct device_aggregate_batch* batch,
                      CUstream metadata_stream);

// Resolves and submits payload spans, performs one host readiness wait, then
// returns normalized per-run views into the pinned slot.
int
d2h_materialize_finish(struct d2h_materializer* materializer,
                       int slot_index,
                       const struct d2h_host_placement* placement,
                       struct host_batch** out);
