#pragma once

#include "stream/dim_info.h"
#include "stream/layouts.h"
#include "stream/types.aggregate.h"
#include "writer.h"
#include <stddef.h>
#include <stdint.h>

// A shard file ends with one page-aligned write — its **footer** —
// containing [<page trailing chunk bytes || index || crc || zero-pad].
// The footer combines the ragged sub-page leftover after the page-floor
// write with the index/crc so close-out is a single O_DIRECT call.

struct active_shard
{
  uint64_t data_cursor;        // file offset for next write
  uint64_t* index;             // [2 * chunks_per_shard_total]
  struct shard_writer* writer; // sink->open(), NULL between generations

  // Sub-page bytes carried across batches (leading tail prepended to the
  // next batch's first run). Slice of shard_state.tail_buf_pool. tail_bytes
  // (< page) is the validity gate; tail_buf may retain stale bytes after
  // finalize zeros tail_bytes.
  uint8_t* tail_buf;
  size_t tail_bytes;

  // Slice of shard_state.footer_buf_pool, lifetime-fenced via footer_io_done.
  // Reuse rule: wait_fence(footer_io_done) before refilling, record_fence
  // after every write_direct.
  uint8_t* footer_buf;
  struct io_event footer_io_done;
};

struct shard_state
{
  uint64_t epoch_in_shard;    // 0..chunks_per_shard_append-1
  uint64_t shard_epoch;       // flat append shard index
  uint64_t shard_inner_count; // S_inner = prod(shard_count[d>=n_append])
  uint64_t chunks_per_shard_inner;
  uint64_t chunks_per_shard_total;
  uint64_t chunks_per_shard_append;
  struct active_shard* shards; // [shard_inner_count]

  // Append chunks in shards closed out with their index block written, so a
  // reader can parse them. Grows only at finalize.
  uint64_t finalized_append_chunks;
  struct io_event finalized_fence;
  int fence_pending; // finalized_fence not waited on yet

  // Contiguous tail pool, layout matches GPU's d_tail_carry so a single
  // bulk HtoD upload covers all shards. NULL when page == 0.
  uint8_t* tail_buf_pool;
  size_t tail_buf_pool_bytes;

  // Contiguous footer pool (page-aligned). NULL when page == 0.
  uint8_t* footer_buf_pool;
  size_t footer_buf_pool_bytes;
  size_t footer_capacity; // bytes per shard

  // Most a shard file can reach: every chunk at its worst-case compressed
  // size, plus the footer. Zero when the size is unknown, which turns
  // pre-sizing off.
  uint64_t shard_file_capacity;
};

int
init_shard_state(struct shard_state* ss, const struct level_layout_info* li);

// Safe on partially-initialized or zeroed state; leaves *ss zeroed.
void
shard_state_destroy(struct shard_state* ss);

// Sizing mirror of init_shard_state, for the memory estimate.
size_t
shard_state_heap_bytes(const struct level_layout_info* li);

// A reader can safely see up to this append-dim extent. The last finalize's
// writes are waited on, and they have normally landed already.
uint64_t
shard_state_readable_append_chunks(struct shard_state* ss,
                                   struct shard_sink* sink);

// Publish one level's append extent through the sink. Pass cursor_elements to
// hold the extent down to what the caller appended, or NULL where the cursor
// belongs to another thread. Returns non-zero if the sink rejected the update.
//
// The extent names only closed-out shards, so it stays truthful after a failed
// flush, and is the only way a reader learns of shards written since the last
// periodic update.
int
shard_state_publish_append(struct shard_state* ss,
                           struct shard_sink* sink,
                           const struct dim_info* dims,
                           uint8_t level,
                           const uint64_t* cursor_elements);

// Best-effort finalize of every shard with an open writer. Returns 0 on
// success. Calls sink->wait_fence/record_fence on each shard's footer to
// keep the footer_buf reuse cycle correct.
int
finalize_shards(struct shard_state* ss,
                struct shard_sink* sink,
                size_t shard_alignment);

// Deliver compressed chunks from one batch's aggregate slot to shards.
//   layout: aggregate layout for shard_capacity / num_shards / page_size.
//   h_tail_bytes: [num_shards] sub-page leading-tail bytes carried in;
//                 updated in place. NULL only when page_size == 0.
int
deliver_to_shards_batch(uint8_t level,
                        struct shard_state* ss,
                        struct aggregate_result* result,
                        const struct aggregate_layout* layout,
                        size_t* h_tail_bytes,
                        uint32_t n_active,
                        struct shard_sink* sink,
                        size_t shard_alignment,
                        size_t* out_bytes);
