#include "zarr/shard_delivery.h"

#include "defs.limits.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"
#include "zarr/crc32c.h"

#include <stdlib.h>
#include <string.h>

// Wait for every write queued up to the event, timed into the metric when one
// is given.
static void
wait_fence_timed(struct shard_sink* sink,
                 struct io_event ev,
                 struct stream_metric* metric)
{
  if (!sink || !sink->wait_fence)
    return;
  if (!metric) {
    sink->wait_fence(sink, ev);
    return;
  }
  struct platform_clock clk = { 0 };
  platform_toc(&clk);
  sink->wait_fence(sink, ev);
  accumulate_metric_ms(metric, (float)(platform_toc(&clk) * 1000.0), 0, 0);
}

// Zero when the product would not fit, which turns pre-sizing off rather
// than pre-sizing to the wrong number.
static uint64_t
shard_file_capacity_for(uint64_t chunks_per_shard_total,
                        size_t max_comp_chunk_bytes,
                        size_t footer_capacity)
{
  if (max_comp_chunk_bytes == 0 ||
      chunks_per_shard_total >
        (UINT64_MAX - footer_capacity) / max_comp_chunk_bytes)
    return 0;
  return chunks_per_shard_total * max_comp_chunk_bytes + footer_capacity;
}

int
init_shard_state(struct shard_state* ss, const struct level_layout_info* li)
{
  *ss = (struct shard_state){
    .chunks_per_shard_append = li->chunks_per_shard_append,
    .chunks_per_shard_inner = li->chunks_per_shard_inner,
    .chunks_per_shard_total = li->chunks_per_shard_total,
    .shard_inner_count = li->shard_inner_count,
  };
  ss->shards = (struct active_shard*)calloc(li->shard_inner_count,
                                            sizeof(struct active_shard));
  if (!ss->shards)
    return 1;
  const size_t page = li->agg_layout.page_size;
  if (page > 0 && li->shard_inner_count > 0) {
    ss->tail_buf_pool_bytes = (size_t)li->shard_inner_count * page;
    ss->tail_buf_pool = (uint8_t*)calloc(1, ss->tail_buf_pool_bytes);
    if (!ss->tail_buf_pool)
      return 1;
    ss->footer_capacity = footer_capacity_for(li->chunks_per_shard_total, page);
    if (ss->footer_capacity == 0)
      return 1;
    ss->footer_buf_pool_bytes =
      (size_t)li->shard_inner_count * ss->footer_capacity;
    ss->footer_buf_pool =
      (uint8_t*)platform_aligned_alloc(page, ss->footer_buf_pool_bytes);
    if (!ss->footer_buf_pool)
      return 1;
    ss->shard_file_capacity =
      shard_file_capacity_for(li->chunks_per_shard_total,
                              li->agg_layout.max_comp_chunk_bytes,
                              ss->footer_capacity);
  }
  for (uint64_t si = 0; si < li->shard_inner_count; ++si) {
    ss->shards[si].index =
      (uint64_t*)malloc(li->chunks_per_shard_total * 2 * sizeof(uint64_t));
    if (!ss->shards[si].index)
      return 1;
    memset(ss->shards[si].index,
           0xFF,
           li->chunks_per_shard_total * 2 * sizeof(uint64_t));
    if (page > 0) {
      ss->shards[si].tail_buf = ss->tail_buf_pool + si * page;
      ss->shards[si].footer_buf =
        ss->footer_buf_pool + si * ss->footer_capacity;
    }
  }
  return 0;
}

// Host heap bytes init_shard_state allocates for this level. Must mirror
// the allocations above exactly — tile_stream_gpu_memory_estimate sums this.
size_t
shard_state_heap_bytes(const struct level_layout_info* li)
{
  size_t bytes = li->shard_inner_count * sizeof(struct active_shard);
  const size_t page = li->agg_layout.page_size;
  if (page > 0 && li->shard_inner_count > 0) {
    bytes += (size_t)li->shard_inner_count * page; // tail_buf_pool
    bytes += (size_t)li->shard_inner_count *
             footer_capacity_for(li->chunks_per_shard_total, page);
  }
  bytes +=
    li->shard_inner_count * (li->chunks_per_shard_total * 2 * sizeof(uint64_t));
  return bytes;
}

void
shard_state_destroy(struct shard_state* ss)
{
  if (ss->shards) {
    for (uint64_t si = 0; si < ss->shard_inner_count; ++si)
      free(ss->shards[si].index);
    free(ss->shards);
  }
  free(ss->tail_buf_pool);
  if (ss->footer_buf_pool)
    platform_aligned_free(ss->footer_buf_pool);
  *ss = (struct shard_state){ 0 };
}

// Build [tail || index || crc || zero-pad] into dst. Writes aligned_bytes
// total (= logical_bytes rounded up to shard_alignment); the caller passes
// dst_capacity >= aligned_bytes.
static int
build_shard_footer(uint8_t* dst,
                   size_t dst_capacity,
                   const uint8_t* tail_src,
                   size_t tail_bytes,
                   const uint64_t* index_data,
                   size_t index_data_bytes,
                   size_t shard_alignment,
                   size_t* out_aligned_bytes,
                   size_t* out_logical_bytes)
{
  size_t logical_bytes = tail_bytes + index_data_bytes + 4;
  size_t aligned_bytes = shard_alignment > 0
                           ? align_up(logical_bytes, shard_alignment)
                           : logical_bytes;
  if (aligned_bytes > dst_capacity)
    return 1;

  if (tail_bytes > 0)
    memcpy(dst, tail_src, tail_bytes);
  memcpy(dst + tail_bytes, index_data, index_data_bytes);
  uint32_t crc_val = crc32c(dst + tail_bytes, index_data_bytes);
  memcpy(dst + tail_bytes + index_data_bytes, &crc_val, 4);
  if (aligned_bytes > logical_bytes)
    memset(dst + logical_bytes, 0, aligned_bytes - logical_bytes);

  *out_aligned_bytes = aligned_bytes;
  *out_logical_bytes = logical_bytes;
  return 0;
}

// Build sh->footer_buf and write it via write_direct, fenced on
// sh->footer_io_done. On sinks without write_direct (e.g. S3) bounce-copy
// through a malloc'd buffer instead.
//
// Returns 0 on success; on failure logs and returns 1. *out_logical_bytes
// holds the unpadded footer size (used for the truncate target).
static int
write_footer(struct active_shard* sh,
             const struct shard_state* ss,
             struct shard_sink* sink,
             const uint8_t* tail_src,
             size_t tail_bytes,
             size_t shard_alignment,
             size_t* out_aligned_bytes,
             size_t* out_logical_bytes,
             struct stream_metrics* metrics)
{
  const size_t index_data_bytes =
    ss->chunks_per_shard_total * 2 * sizeof(uint64_t);

  if (sh->footer_buf && sh->writer->write_direct) {
    wait_fence_timed(
      sink, sh->footer_io_done, metrics ? &metrics->footer_buffer_stall : NULL);
    if (build_shard_footer(sh->footer_buf,
                           ss->footer_capacity,
                           tail_src,
                           tail_bytes,
                           sh->index,
                           index_data_bytes,
                           shard_alignment,
                           out_aligned_bytes,
                           out_logical_bytes))
      return 1;
    if (sh->writer->write_direct(sh->writer,
                                 sh->data_cursor,
                                 sh->footer_buf,
                                 sh->footer_buf + *out_aligned_bytes))
      return 1;
    if (sink && sink->record_fence)
      sh->footer_io_done = sink->record_fence(sink);
    return 0;
  }

  // Bounce path: write() copies into the job; buffer can be freed immediately.
  size_t cap = tail_bytes + index_data_bytes + 4;
  if (shard_alignment > 0)
    cap = align_up(cap, shard_alignment);
  uint8_t* buf = shard_alignment > 0
                   ? (uint8_t*)platform_aligned_alloc(shard_alignment, cap)
                   : (uint8_t*)malloc(cap);
  if (!buf)
    return 1;
  int err = build_shard_footer(buf,
                               cap,
                               tail_src,
                               tail_bytes,
                               sh->index,
                               index_data_bytes,
                               shard_alignment,
                               out_aligned_bytes,
                               out_logical_bytes) ||
            sh->writer->write(
              sh->writer, sh->data_cursor, buf, buf + *out_aligned_bytes);
  if (shard_alignment > 0)
    platform_aligned_free(buf);
  else
    free(buf);
  return err;
}

// The shards' reach along the append dim is recorded, not the number of chunks
// they hold: a flush that closes a half-full shard leaves its remaining slots
// empty and the next generation still starts after them.
static void
record_finalized(struct shard_state* ss, struct shard_sink* sink)
{
  ss->finalized_append_chunks =
    ss->shard_epoch * ss->chunks_per_shard_append + ss->epoch_in_shard;
  if (sink->record_fence) {
    ss->finalized_fence = sink->record_fence(sink);
    ss->fence_pending = 1;
  }
}

uint64_t
shard_state_readable_append_chunks(struct shard_state* ss,
                                   struct shard_sink* sink,
                                   struct stream_metrics* metrics)
{
  // Once per closed-out generation, not once per caller: the metadata update
  // runs every batch, and re-waiting a retired fence costs real throughput on
  // small epochs.
  if (ss->fence_pending) {
    wait_fence_timed(sink,
                     ss->finalized_fence,
                     metrics ? &metrics->append_extent_stall : NULL);
    ss->fence_pending = 0;
  }
  return ss->finalized_append_chunks;
}

int
shard_state_publish_append(struct shard_state* ss,
                           struct shard_sink* sink,
                           const struct dim_info* dims,
                           uint8_t level,
                           const uint64_t* cursor_elements,
                           struct stream_metrics* metrics)
{
  uint64_t readable = shard_state_readable_append_chunks(ss, sink, metrics);
  uint64_t append_sizes[HALF_MAX_RANK];
  if (cursor_elements)
    dim_info_readable_append_sizes(
      dims, readable, *cursor_elements, level, append_sizes);
  else
    dim_info_decompose_append_sizes(dims, readable, append_sizes);
  return sink->update_append(
    sink, level, dim_info_n_append(dims), append_sizes);
}

int
finalize_shards(struct shard_state* ss,
                struct shard_sink* sink,
                size_t shard_alignment,
                struct stream_metrics* metrics)
{
  int err = 0;
  size_t index_data_bytes = ss->chunks_per_shard_total * 2 * sizeof(uint64_t);

  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    size_t aligned_bytes = 0;
    size_t logical_bytes = 0;
    if (write_footer(sh,
                     ss,
                     sink,
                     sh->tail_buf,
                     sh->tail_bytes,
                     shard_alignment,
                     &aligned_bytes,
                     &logical_bytes,
                     metrics)) {
      log_error("finalize_shards: footer write failed for shard %llu",
                (unsigned long long)si);
      err = 1;
    }

    if (!err && sh->writer->truncate) {
      uint64_t logical_size = sh->data_cursor + logical_bytes;
      if (sh->writer->truncate(sh->writer, logical_size)) {
        log_error("finalize_shards: truncate failed for shard %llu",
                  (unsigned long long)si);
        err = 1;
      }
    }

    if (sh->writer->finalize(sh->writer)) {
      log_error("finalize_shards: finalize failed for shard %llu",
                (unsigned long long)si);
      err = 1;
    }

    sh->writer = NULL;
    sh->data_cursor = 0;
    sh->tail_bytes = 0;
    memset(sh->index, 0xFF, index_data_bytes);
  }

  if (!err)
    record_finalized(ss, sink);
  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return err;
}
