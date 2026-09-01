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
struct stream_metric;

enum aggregate_size_kind
{
  AGGREGATE_FIXED_SIZE = 0,
  AGGREGATE_VARIABLE_SIZE,
};

struct aggregate_batch
{
  int slot_index;
  enum aggregate_size_kind size_kind;
  uint32_t epoch_count;
  struct batch_aggregate_layout layout;
  uint32_t active_count_by_level[LOD_MAX_LEVELS];
  const struct aggregate_layout* level_layouts;
  size_t fixed_chunk_bytes;
  struct gpu_pool* aggregate_pool;
  struct gpu_pool* host_pool;
  struct gpu_pool* index_pool;
};

enum d2h_copy_status
{
  D2H_COPY_EMPTY = 0,
  D2H_COPY_METADATA_PENDING,
  D2H_COPY_PAYLOAD_PENDING,
  D2H_COPY_HOST_READY,
  D2H_COPY_ERROR,
};

struct d2h_copy_state
{
  struct aggregate_batch batch;
  struct aggregate_slot* slot;
  struct d2h_transfer_span* spans;
  size_t span_count;
  size_t span_capacity;
  struct host_batch host;
  CUevent payload_start;
  int aggregate_acquired;
  int aggregate_released;
  enum d2h_copy_status status;
};

struct host_batch_copy
{
  enum aggregate_size_kind size_kind;
  struct gpu_ordering* ordering;
  CUstream payload_stream;
  CUevent metadata_copy_start[2];
  CUevent payload_event[2];
  struct stream_metric* aggregate_wait;
  struct stream_metric* metadata_wait;
  struct d2h_copy_state state[2];
};

int
host_batch_copy_init(struct host_batch_copy* copy,
                     enum aggregate_size_kind size_kind,
                     struct gpu_ordering* ordering,
                     CUstream payload_stream,
                     CUstream seed_stream);

void
host_batch_copy_set_wait_metrics(struct host_batch_copy* copy,
                                 struct stream_metric* aggregate_wait,
                                 struct stream_metric* metadata_wait);

void
host_batch_copy_destroy(struct host_batch_copy* copy);

int
host_batch_copy_begin(struct host_batch_copy* copy,
                      const struct aggregate_batch* batch,
                      CUstream metadata_stream);

int
host_batch_copy_finish(struct host_batch_copy* copy,
                       int slot_index,
                       struct shard_state* const* shards_by_level,
                       size_t shard_alignment,
                       struct host_batch** out);

int
host_batch_copy_cancel(struct host_batch_copy* copy, int slot_index);
