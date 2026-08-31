#pragma once

#include "zarr/shard_delivery.h"

#include <stddef.h>
#include <stdint.h>

// Pull-style, CUDA-free delivery plan.  A command is immutable until the sink
// accepts it; shard_drain_accept is the only operation that commits its state
// transition.  abort is sticky and deliberately does not roll back commands
// the sink already accepted.
enum shard_drain_command_kind
{
  SHARD_DRAIN_DATA = 0,
  SHARD_DRAIN_FOOTER,
  SHARD_DRAIN_TRUNCATE,
  SHARD_DRAIN_FINALIZE,
};

enum shard_drain_buffer_lease_kind
{
  SHARD_DRAIN_LEASE_NONE = 0,
  SHARD_DRAIN_LEASE_HOST_BATCH,
  SHARD_DRAIN_LEASE_FOOTER,
  SHARD_DRAIN_LEASE_TRANSIENT,
};

struct shard_drain_buffer_lease
{
  enum shard_drain_buffer_lease_kind kind;
  void* owner;
};

struct shard_drain_command
{
  enum shard_drain_command_kind kind;
  uint64_t serial;

  uint8_t level;
  uint64_t inner_shard;
  uint64_t flat_shard;
  uint64_t file_offset;

  const uint8_t* source_begin;
  const uint8_t* source_end;
  size_t physical_bytes;
  uint64_t logical_size; // truncate target; zero for other commands
  int direct_write_eligible;
  struct shard_drain_buffer_lease buffer_lease;

  // Write-layout accounting is attached to exactly one accepted command for
  // each nonempty physical-shard update, independent of how many data/footer
  // commands the update requires.
  size_t logical_payload_bytes;
  size_t internal_padding_bytes;
  int counts_physical_update;
  int counts_padded_update;

  // True only for the final FINALIZE command across all inner shards in one
  // generation run.  The executor records the sink fence after accepting it.
  int closes_generation;
};

enum shard_drain_phase
{
  SHARD_DRAIN_PHASE_RUN = 0,
  SHARD_DRAIN_PHASE_DATA,
  SHARD_DRAIN_PHASE_FOOTER,
  SHARD_DRAIN_PHASE_TRUNCATE,
  SHARD_DRAIN_PHASE_FINALIZE,
};

struct shard_drainer
{
  struct host_batch* host;
  struct shard_state* const* shards_by_lod;
  size_t shard_alignment;
  size_t run_index;
  enum shard_drain_phase phase;

  uint64_t next_serial;
  int pending;
  int prepared;
  int failed;

  uint64_t run_start_cursor;
  size_t data_physical_bytes;
  size_t footer_remainder_bytes;
  size_t footer_logical_bytes;
  size_t footer_physical_bytes;
  int metric_recorded;

  uint8_t* transient_footer;
  struct shard_drain_command current;
};

// Begin from the currently committed shard state. Returns non-zero on invalid
// input. The host batch and shard-state array are borrowed until destroy.
int
shard_drain_begin(struct shard_drainer* drain,
                  struct host_batch* host,
                  struct shard_state* const* shards_by_lod);

// 1: one command returned; 0: complete; -1: sticky failure.
int
shard_drain_next(struct shard_drainer* drain,
                 struct shard_drain_command* command);

// FOOTER commands are prepared only after the thin sink executor has waited
// for command.buffer_lease. Other command kinds are already prepared and this
// is a no-op.
int
shard_drain_prepare(struct shard_drainer* drain,
                    struct shard_drain_command* command);

// Commit the pending command after the sink accepts it.
int
shard_drain_accept(struct shard_drainer* drain,
                   const struct shard_drain_command* command);

// Mark this stream sticky-failed. Previously accepted state remains committed.
void
shard_drain_abort(struct shard_drainer* drain);

void
shard_drain_destroy(struct shard_drainer* drain);
