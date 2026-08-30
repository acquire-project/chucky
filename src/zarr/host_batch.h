#pragma once

#include <stddef.h>
#include <stdint.h>

struct batch_aggregate_layout;

struct d2h_transfer_span
{
  size_t device_offset;
  size_t host_offset;
  size_t bytes;
};

// Statistics owned by materialization rather than sink delivery.  Logical
// bytes exclude carried tails and padding; transferred bytes are the actual
// payload bytes copied from device memory.
struct d2h_transfer_statistics
{
  size_t logical_payload_bytes;
  size_t payload_bytes_transferred;
  size_t metadata_bytes_transferred;
  size_t payload_copy_count;
};

// One physical shard's portion of one append-generation run.  Offsets remain
// absolute in the device aggregate; source_offset is their normalization
// origin.  data points at the run's leading host-tail prefix, if any.
struct host_batch_run
{
  uint8_t level;
  uint64_t inner_shard;
  uint64_t flat_shard;

  uint32_t active_begin;
  uint32_t active_count;
  uint64_t epoch_in_shard;
  uint64_t chunks_per_shard_inner;

  int finalizes;
  int ends_generation_run;

  uint8_t* data;
  size_t page_size;
  size_t tail_bytes;
  size_t payload_bytes;
  size_t source_offset;
  const size_t* offsets;
  const size_t* chunk_sizes;
};

// Normalized materialized batch.  slot_lifetime is deliberately opaque: the
// GPU schedule owns the pool lease and sink fence that keep these views alive.
struct host_batch
{
  struct host_batch_run* runs;
  size_t run_count;
  size_t run_capacity;
  void* slot_lifetime;
  struct d2h_transfer_statistics transfer;
};

// Behavior-preserving span planner for the legacy device layout.  Fixed
// extents copy the reserved aggregate as one span.  Indexed extents use the
// landed metadata to trim each LOD (or the one contiguous aggregate).
int
d2h_plan_legacy_spans(const struct batch_aggregate_layout* layout,
                      uint8_t nlod,
                      const uint32_t* per_lod_n_active,
                      const size_t* offsets,
                      const size_t* chunk_sizes,
                      int fixed_extent,
                      struct d2h_transfer_span* spans,
                      size_t span_capacity,
                      size_t* out_count);
