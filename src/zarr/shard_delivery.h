#pragma once

#include "stream/layouts.h"
#include "stream/types.aggregate.h"
#include "writer.h"
#include <stddef.h>
#include <stdint.h>

struct active_shard
{
  size_t data_cursor;
  uint64_t* index;             // 2 * chunks_per_shard_total entries
  struct shard_writer* writer; // from sink->open, NULL until first use
};

struct shard_state
{
  uint64_t epoch_in_shard;         // 0..chunks_per_shard[0]-1
  uint64_t shard_epoch;            // s_0 coordinate (0, 1, 2, ...)
  uint64_t shard_inner_count;      // S_inner = prod(shard_count[d] for d>0)
  uint64_t chunks_per_shard_inner; // prod(tps[d] for d>0)
  uint64_t chunks_per_shard_total; // prod(tps[d] for all d)
  uint64_t chunks_per_shard_append;     // tps[0]
  struct active_shard* shards;     // array[shard_inner_count]
};

// Initialize shard state from pre-computed level layout info.
// Allocates the shards array and per-shard index buffers.
int
init_shard_state(struct shard_state* ss,
                 const struct level_layout_info* li);

// Finalize completed shards (write index block + finalize).
// Best-effort: tries every shard even if one fails. Returns 0 on success.
int
finalize_shards(struct shard_state* ss, size_t shard_alignment);

// Deliver compressed chunk data from a batch aggregate slot to shards.
// n_active: number of active epochs for this level in the batch.
int
deliver_to_shards_batch(uint8_t level,
                        struct shard_state* ss,
                        struct aggregate_result* result,
                        uint32_t n_active,
                        struct shard_sink* sink,
                        size_t shard_alignment,
                        size_t* out_bytes);
