#pragma once

#include <stddef.h>
#include <stdint.h>

struct aggregate_layout;
struct batch_aggregate_layout;
struct shard_state;
struct host_output_group;

enum host_batch_storage
{
  HOST_BATCH_FIXED_SIZE = 0,
  HOST_BATCH_PAGE_PADDED,
  HOST_BATCH_PACKED,
};

enum host_batch_storage
host_batch_storage_select(int fixed_size, size_t shard_alignment);

struct d2h_transfer_span
{
  size_t device_offset;
  size_t host_offset;
  size_t bytes;
};

struct d2h_transfer_statistics
{
  size_t payload_bytes_transferred;
  size_t metadata_bytes_transferred;
  size_t payload_copy_count;
};

struct host_batch_run
{
  uint8_t level;
  uint64_t inner_shard;
  uint64_t flat_shard;
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

struct host_batch
{
  struct host_batch_run* runs;
  size_t run_count;
  size_t run_capacity;
  uint8_t nlod;
  enum host_batch_storage storage;
  size_t shard_alignment;
  struct host_output_group* output_group;
  struct d2h_transfer_statistics transfer;
};

int
host_batch_capacity(const struct aggregate_layout* level_layouts,
                    const uint32_t* active_count_by_level,
                    uint8_t nlod,
                    enum host_batch_storage storage,
                    size_t shard_alignment,
                    size_t* out_bytes,
                    size_t* out_run_count);

int
host_batch_build(struct host_batch* host,
                 void* aggregate_data,
                 size_t aggregate_capacity,
                 const size_t* offsets,
                 const size_t* chunk_sizes,
                 const struct batch_aggregate_layout* batch_layout,
                 const struct aggregate_layout* level_layouts,
                 struct shard_state* const* shards_by_level,
                 const uint32_t* active_count_by_level,
                 enum host_batch_storage storage,
                 size_t shard_alignment,
                 struct d2h_transfer_span* spans,
                 size_t span_capacity,
                 size_t* out_span_count);

void
host_batch_destroy(struct host_batch* host);
