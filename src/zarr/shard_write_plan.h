#pragma once

#include "zarr/shard_delivery.h"

#include <stddef.h>
#include <stdint.h>

enum shard_write_kind
{
  SHARD_WRITE_DATA = 0,
  SHARD_WRITE_FOOTER,
  SHARD_WRITE_TRUNCATE,
  SHARD_WRITE_FINALIZE,
};

struct shard_write_command
{
  enum shard_write_kind kind;
  uint64_t serial;
  uint8_t level;
  uint64_t inner_shard;
  uint64_t flat_shard;
  uint64_t file_offset;
  const uint8_t* source;
  size_t write_size;
  uint64_t truncate_size;
  size_t payload_bytes;
  size_t padding_bytes;
  int counts_shard_update;
  int counts_padded_update;
  int closes_generation;
};

enum shard_write_phase
{
  SHARD_WRITE_PHASE_RUN = 0,
  SHARD_WRITE_PHASE_DATA,
  SHARD_WRITE_PHASE_FOOTER,
  SHARD_WRITE_PHASE_TRUNCATE,
  SHARD_WRITE_PHASE_FINALIZE,
};

struct shard_write_plan
{
  struct host_batch* host;
  struct shard_state* const* shards_by_level;
  size_t shard_alignment;
  size_t run_index;
  enum shard_write_phase phase;
  uint64_t next_serial;
  int pending;
  int prepared;
  int failed;
  uint64_t run_start_cursor;
  size_t data_physical_bytes;
  size_t footer_remainder_bytes;
  size_t footer_logical_bytes;
  size_t footer_physical_bytes;
  uint8_t* transient_footer;
  struct shard_write_command current;
};

int
shard_write_begin(struct shard_write_plan* plan,
                  struct host_batch* host,
                  struct shard_state* const* shards_by_level);

int
shard_write_next(struct shard_write_plan* plan,
                 struct shard_write_command* command);

int
shard_write_prepare(struct shard_write_plan* plan,
                    struct shard_write_command* command);

int
shard_write_accept(struct shard_write_plan* plan,
                   const struct shard_write_command* command);

void
shard_write_abort(struct shard_write_plan* plan);

void
shard_write_destroy(struct shard_write_plan* plan);

int
deliver_host_batch(struct host_batch* host,
                   struct shard_state* const* shards_by_level,
                   struct shard_sink* sink,
                   size_t* out_bytes,
                   struct stream_metrics* metrics);
