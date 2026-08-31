#include "zarr/shard_delivery.h"

#include "defs.limits.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "util/prelude.h"
#include "zarr/crc32c.h"

#include <stdlib.h>
#include <string.h>

enum host_delivery_policy
host_delivery_policy_select(int fixed_extent, size_t shard_alignment)
{
  if (fixed_extent)
    return HOST_DELIVERY_FIXED_TAIL;
  return shard_alignment > 0 ? HOST_DELIVERY_INDEXED_PADDED
                             : HOST_DELIVERY_INDEXED_COMPACT;
}

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

static void
shard_tail_set(struct active_shard* sh,
               size_t* h_tail_bytes_si,
               const uint8_t* src,
               size_t n)
{
  sh->tail_bytes = n;
  if (h_tail_bytes_si)
    *h_tail_bytes_si = n;
  if (n > 0)
    memcpy(sh->tail_buf, src, n);
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

// Sum chunk_sizes for chunks in shard si over epoch range [a, a+run_len).
// Used in the legacy shard-capacity path instead of
// `offsets[j_run_end] - offsets[j_run_start]`: each shard's local offsets do
// not guarantee that a cross-region sentinel forms a valid difference.
static size_t
sum_run_chunks(const size_t* chunk_sizes,
               uint64_t si,
               uint32_t a,
               uint32_t run_len,
               uint32_t n_active,
               uint64_t cps_inner)
{
  uint64_t base = si * (uint64_t)n_active * cps_inner + (uint64_t)a * cps_inner;
  size_t sum = 0;
  for (uint64_t k = 0; k < (uint64_t)run_len * cps_inner; ++k)
    sum += chunk_sizes[base + k];
  return sum;
}

// For each chunk j in [j_run_start, j_run_start + run_len*cps_inner),
// set sh->index[2*slot] = data_cursor + (offsets[j] - base_off).
static void
record_run_index(struct active_shard* sh,
                 const struct aggregate_result* result,
                 uint64_t j_run_start,
                 size_t base_off,
                 uint32_t run_len,
                 uint64_t eis_start,
                 uint64_t cps_inner)
{
  for (uint32_t r = 0; r < run_len; ++r) {
    uint64_t eis = eis_start + r;
    uint64_t j_start = j_run_start + (uint64_t)r * cps_inner;
    for (uint64_t j = j_start; j < j_start + cps_inner; ++j) {
      size_t chunk_size = result->chunk_sizes[j];
      if (chunk_size == 0)
        continue;
      uint64_t within_inner = j - j_start;
      uint64_t slot_idx = eis * cps_inner + within_inner;
      sh->index[2 * slot_idx] =
        sh->data_cursor + (result->offsets[j] - base_off);
      sh->index[2 * slot_idx + 1] = chunk_size;
    }
  }
}

// Page-floor write of run data, then footer write that closes the shard.
static int
deliver_run_finalizing(struct active_shard* sh,
                       const struct shard_state* ss,
                       struct shard_sink* sink,
                       const uint8_t* src,
                       size_t total_run,
                       size_t* h_tail_bytes_si,
                       size_t sa,
                       size_t* total_bytes,
                       struct stream_metrics* metrics)
{
  size_t page_floor = sa > 0 ? (total_run / sa) * sa : 0;
  if (page_floor > 0) {
    int aligned = ((uintptr_t)src % sa == 0);
    int wr =
      (aligned && sh->writer->write_direct)
        ? sh->writer->write_direct(
            sh->writer, sh->data_cursor, src, src + page_floor)
        : sh->writer->write(sh->writer, sh->data_cursor, src, src + page_floor);
    if (wr)
      return 1;
    *total_bytes += page_floor;
    sh->data_cursor += page_floor;
  }

  size_t aligned_bytes = 0;
  size_t logical_bytes = 0;
  if (write_footer(sh,
                   ss,
                   sink,
                   src + page_floor,
                   total_run - page_floor,
                   sa,
                   &aligned_bytes,
                   &logical_bytes,
                   metrics))
    return 1;
  *total_bytes += aligned_bytes;

  if (sh->writer->truncate) {
    uint64_t logical_size = sh->data_cursor + logical_bytes;
    if (sh->writer->truncate(sh->writer, logical_size))
      return 1;
  }

  shard_tail_set(sh, h_tail_bytes_si, NULL, 0);
  return 0;
}

// Page-floor write; sub-page remainder rolls into next batch's leading tail.
static int
deliver_run_nonfinalizing(struct active_shard* sh,
                          const uint8_t* src,
                          size_t total_run,
                          size_t* h_tail_bytes_si,
                          size_t page_size,
                          size_t sa,
                          size_t* total_bytes)
{
  size_t write_bytes = (total_run / page_size) * page_size;
  if (write_bytes > 0) {
    const uint8_t* src_end = src + write_bytes;
    int aligned = ((uintptr_t)src % sa == 0);
    int wr =
      (aligned && sh->writer->write_direct)
        ? sh->writer->write_direct(sh->writer, sh->data_cursor, src, src_end)
        : sh->writer->write(sh->writer, sh->data_cursor, src, src_end);
    if (wr)
      return 1;
    *total_bytes += write_bytes;
  }
  sh->data_cursor += write_bytes;
  shard_tail_set(
    sh, h_tail_bytes_si, src + write_bytes, total_run - write_bytes);
  return 0;
}

// Legacy path (no alignment requirement): one write per run.
static int
deliver_run_contiguous(struct active_shard* sh,
                       const struct aggregate_result* result,
                       uint64_t j_run_start,
                       uint64_t j_run_end,
                       uint32_t run_len,
                       uint64_t eis_start,
                       uint64_t cps_inner,
                       size_t sa,
                       size_t* total_bytes)
{
  size_t run_bytes = result->offsets[j_run_end] - result->offsets[j_run_start];
  if (run_bytes > 0) {
    const void* src = (const char*)result->data + result->offsets[j_run_start];
    size_t write_bytes = sa > 0 ? align_up(run_bytes, sa) : run_bytes;
    const void* src_end = (const char*)src + write_bytes;
    int aligned = sa == 0 || ((uintptr_t)src % sa == 0);
    int wr =
      (aligned && sh->writer->write_direct)
        ? sh->writer->write_direct(sh->writer, sh->data_cursor, src, src_end)
        : sh->writer->write(sh->writer, sh->data_cursor, src, src_end);
    if (wr)
      return 1;
    *total_bytes += write_bytes;
  }
  record_run_index(sh,
                   result,
                   j_run_start,
                   result->offsets[j_run_start],
                   run_len,
                   eis_start,
                   cps_inner);
  sh->data_cursor += sa > 0 ? align_up(run_bytes, sa) : run_bytes;
  return 0;
}

// Footer write already emitted the index; just close the writer and reset
// per-shard state for the next generation.
static int
close_finalized_shards(struct shard_state* ss, struct shard_sink* sink)
{
  size_t index_data_bytes = ss->chunks_per_shard_total * 2 * sizeof(uint64_t);
  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;
    if (sh->writer->finalize(sh->writer)) {
      log_error("deliver: finalize failed for shard %llu",
                (unsigned long long)si);
      return 1;
    }
    sh->writer = NULL;
    sh->data_cursor = 0;
    memset(sh->index, 0xFF, index_data_bytes);
  }
  record_finalized(ss, sink);
  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return 0;
}

int
deliver_to_shards_batch(uint8_t level,
                        struct shard_state* ss,
                        struct aggregate_result* result,
                        const struct aggregate_layout* layout,
                        size_t* h_tail_bytes,
                        uint32_t n_active,
                        struct shard_sink* sink,
                        size_t shard_alignment,
                        size_t* out_bytes,
                        struct stream_metrics* metrics)
{
  const uint64_t cps_inner = ss->chunks_per_shard_inner;
  const size_t sa = shard_alignment;
  const size_t page_size = layout ? layout->page_size : 0;
  const size_t shard_capacity = layout ? layout->shard_capacity : 0;
  const int use_carryover = (page_size > 0 && shard_capacity > 0);
  size_t total_bytes = 0;
  // Per-shard cumulative bytes consumed from agg buffer in this batch;
  // base_off for each run is shard_base + bytes_consumed[si].
  size_t* bytes_consumed = NULL;

  CHECK(Error, !use_carryover || h_tail_bytes != NULL);
  CHECK(Error, !use_carryover || sa == page_size);

  if (use_carryover) {
    bytes_consumed = (size_t*)calloc(ss->shard_inner_count, sizeof(size_t));
    if (!bytes_consumed)
      goto Error;
  }

  uint32_t a = 0;
  while (a < n_active) {
    uint32_t remaining_in_shard =
      (uint32_t)(ss->chunks_per_shard_append - ss->epoch_in_shard);
    uint32_t remaining_in_batch = n_active - a;
    uint32_t run_len = remaining_in_shard < remaining_in_batch
                         ? remaining_in_shard
                         : remaining_in_batch;
    int run_finalizes = (run_len == remaining_in_shard);

    for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
      struct active_shard* sh = &ss->shards[si];

      if (!sh->writer) {
        uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
        sh->writer = sink->open(sink, level, flat);
        CHECK(Error, sh->writer);
        if (sh->writer->presize)
          CHECK(Error,
                sh->writer->presize(sh->writer, ss->shard_file_capacity) == 0);
      }

      uint64_t j_run_start = si * n_active * cps_inner + a * cps_inner;
      uint64_t j_run_end = j_run_start + (uint64_t)run_len * cps_inner;

      if (use_carryover) {
        // Aggregate-result contract: shard si's region starts at
        // (si * shard_capacity) inside result->data; leading tail (if any)
        // sits at the head; first chunk at (+ tail_in). h_tail_bytes[si]
        // is reset to 0 by every finalizing run, so a non-zero value here
        // always means "carry-in from the prior batch's last run".
        const size_t shard_base = (size_t)si * shard_capacity;
        const size_t tail_in = h_tail_bytes[si];
        const size_t run_real = sum_run_chunks(
          result->chunk_sizes, si, a, run_len, n_active, cps_inner);
        const size_t total_run = tail_in + run_real;
        const size_t base_off = shard_base + bytes_consumed[si];
        const uint8_t* src = (const uint8_t*)result->data + base_off;

        record_run_index(sh,
                         result,
                         j_run_start,
                         base_off,
                         run_len,
                         ss->epoch_in_shard,
                         cps_inner);

        if (run_finalizes) {
          CHECK(Error,
                deliver_run_finalizing(sh,
                                       ss,
                                       sink,
                                       src,
                                       total_run,
                                       &h_tail_bytes[si],
                                       sa,
                                       &total_bytes,
                                       metrics) == 0);
        } else {
          CHECK(Error,
                deliver_run_nonfinalizing(sh,
                                          src,
                                          total_run,
                                          &h_tail_bytes[si],
                                          page_size,
                                          sa,
                                          &total_bytes) == 0);
        }

        bytes_consumed[si] += total_run;
      } else {
        CHECK(Error,
              deliver_run_contiguous(sh,
                                     result,
                                     j_run_start,
                                     j_run_end,
                                     run_len,
                                     ss->epoch_in_shard,
                                     cps_inner,
                                     sa,
                                     &total_bytes) == 0);
      }
    }

    ss->epoch_in_shard += run_len;
    a += run_len;

    if (ss->epoch_in_shard >= ss->chunks_per_shard_append) {
      if (use_carryover)
        CHECK(Error, close_finalized_shards(ss, sink) == 0);
      else
        CHECK(Error, finalize_shards(ss, sink, sa, metrics) == 0);
    }
  }

  if (out_bytes)
    *out_bytes = total_bytes;
  free(bytes_consumed);
  return 0;

Error:
  free(bytes_consumed);
  return 1;
}

static int
host_batch_reserve(struct host_batch* host, size_t count)
{
  if (count <= host->run_capacity)
    return 0;
  CHECK_MUL_OVERFLOW(Error, count, sizeof(*host->runs), SIZE_MAX);
  struct host_batch_run* p =
    (struct host_batch_run*)realloc(host->runs, count * sizeof(*host->runs));
  CHECK(Error, p);
  host->runs = p;
  host->run_capacity = count;
  return 0;

Error:
  return 1;
}

int
d2h_plan_legacy_spans(const struct batch_aggregate_layout* layout,
                      uint8_t nlod,
                      const uint32_t* per_lod_n_active,
                      const size_t* offsets,
                      const size_t* chunk_sizes,
                      int fixed_extent,
                      struct d2h_transfer_span* spans,
                      size_t span_capacity,
                      size_t* out_count)
{
  CHECK(Error, layout && per_lod_n_active && spans && out_count);
  CHECK(Error, nlod == layout->nlod);
  *out_count = 0;

  if (fixed_extent) {
    CHECK(Error, span_capacity >= 1);
    spans[0] = (struct d2h_transfer_span){
      .device_offset = 0,
      .host_offset = 0,
      .bytes = layout->total_data_bytes,
    };
    *out_count = 1;
    return 0;
  }

  CHECK(Error, offsets && chunk_sizes);
  if (layout->page_size > 0) {
    for (uint8_t lv = 0; lv < nlod; ++lv) {
      if (per_lod_n_active[lv] == 0)
        continue;
      CHECK(Error, *out_count < span_capacity);
      const struct lod_segment* seg = &layout->lods[lv];
      const uint64_t total = (uint64_t)seg->n_active * seg->covering_count;
      CHECK(Error, total > 0);
      const size_t last = seg->batch_covering_offset + (size_t)lv + total - 1;
      CHECK(Error, offsets[last] <= SIZE_MAX - chunk_sizes[last]);
      const size_t end = offsets[last] + chunk_sizes[last];
      CHECK(Error, end >= seg->data_segment_offset);
      const size_t actual = end - seg->data_segment_offset;
      CHECK(Error, actual <= seg->data_segment_bytes);
      spans[(*out_count)++] = (struct d2h_transfer_span){
        .device_offset = seg->data_segment_offset,
        .host_offset = seg->data_segment_offset,
        .bytes = actual,
      };
    }
  } else if (layout->total_batch_covering > 0) {
    CHECK(Error, span_capacity >= 1);
    const size_t n = (size_t)layout->total_batch_covering + nlod;
    CHECK(Error, offsets[n - 1] <= SIZE_MAX - chunk_sizes[n - 1]);
    const size_t total = offsets[n - 1] + chunk_sizes[n - 1];
    spans[0] = (struct d2h_transfer_span){
      .device_offset = 0,
      .host_offset = 0,
      .bytes = total,
    };
    *out_count = 1;
  }
  return 0;

Error:
  if (out_count)
    *out_count = 0;
  return 1;
}

static int
run_payload_bytes(const size_t* chunk_sizes, uint64_t count, size_t* out_bytes)
{
  size_t bytes = 0;
  for (uint64_t i = 0; i < count; ++i) {
    CHECK(Error, chunk_sizes[i] <= SIZE_MAX - bytes);
    bytes += chunk_sizes[i];
  }
  *out_bytes = bytes;
  return 0;

Error:
  return 1;
}

static int
checked_align_up_size(size_t value, size_t alignment, size_t* out)
{
  CHECK(Error, out && alignment > 0);
  const size_t rem = value % alignment;
  const size_t add = rem == 0 ? 0 : alignment - rem;
  CHECK(Error, value <= SIZE_MAX - add);
  *out = value + add;
  return 0;

Error:
  return 1;
}

static int
maximum_generation_runs(uint32_t active,
                        uint64_t chunks_per_shard_append,
                        uint64_t* out)
{
  CHECK(Error, out && chunks_per_shard_append > 0);
  if (active == 0) {
    *out = 0;
    return 0;
  }
  // The worst committed starting point has one append epoch left in the
  // current generation, followed by full generations.
  const uint64_t left = (uint64_t)active - 1;
  *out =
    1 + left / chunks_per_shard_append + (left % chunks_per_shard_append != 0);
  return 0;

Error:
  return 1;
}

int
host_batch_compact_capacity(const struct aggregate_layout* per_lod_layouts,
                            const uint32_t* per_lod_n_active,
                            uint8_t nlod,
                            enum host_delivery_policy policy,
                            size_t shard_alignment,
                            size_t* out_bytes,
                            size_t* out_run_count)
{
  CHECK(Error, per_lod_layouts && per_lod_n_active);
  CHECK(Error, out_bytes && out_run_count);
  CHECK(Error, nlod >= 1 && nlod <= LOD_MAX_LEVELS);

  size_t bytes = 0;
  size_t runs = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct aggregate_layout* al = &per_lod_layouts[lv];
    const uint32_t active = per_lod_n_active[lv];
    CHECK(Error, al->page_size == shard_alignment);
    CHECK(Error, al->cps_inner > 0 && al->num_shards > 0);
    CHECK_MUL_OVERFLOW(Error, al->num_shards, al->cps_inner, UINT64_MAX);
    CHECK(Error, al->covering_count == al->num_shards * al->cps_inner);
    CHECK(Error, al->chunks_per_shard_append > 0);
    CHECK_MUL_OVERFLOW(Error, active, al->chunks_per_epoch, UINT64_MAX);
    const uint64_t chunks = (uint64_t)active * al->chunks_per_epoch;
    CHECK_MUL_OVERFLOW(Error, chunks, al->max_comp_chunk_bytes, SIZE_MAX);
    const size_t payload = (size_t)chunks * al->max_comp_chunk_bytes;

    uint64_t generation_runs = 0;
    CHECK(Error,
          maximum_generation_runs(
            active, al->chunks_per_shard_append, &generation_runs) == 0);
    CHECK_MUL_OVERFLOW(Error, generation_runs, al->num_shards, SIZE_MAX);
    const size_t physical_runs = (size_t)(generation_runs * al->num_shards);
    CHECK(Error, physical_runs <= SIZE_MAX - runs);
    runs += physical_runs;

    if (shard_alignment == 0) {
      CHECK(Error,
            policy == HOST_DELIVERY_FIXED_TAIL ||
              policy == HOST_DELIVERY_INDEXED_COMPACT);
      CHECK(Error, payload <= SIZE_MAX - bytes);
      bytes += payload;
      continue;
    }

    CHECK(Error, shard_alignment > 0);
    CHECK(Error,
          policy == HOST_DELIVERY_FIXED_TAIL ||
            policy == HOST_DELIVERY_INDEXED_PADDED);

    // Each physical run may start alignment-1 bytes after the previous one.
    // Fixed runs may also prepend at most alignment-1 committed tail bytes;
    // indexed runs may retain at most alignment-1 zero bytes after payload.
    // The same checked bound covers both policies without knowing the
    // committed starting epoch or the per-run payload distribution.
    const size_t per_run_slack = shard_alignment - 1;
    CHECK_MUL_OVERFLOW(Error, physical_runs, per_run_slack, SIZE_MAX);
    const size_t one_slack_each = physical_runs * per_run_slack;
    CHECK(Error, one_slack_each <= SIZE_MAX / 2);
    const size_t slack = one_slack_each * 2;
    CHECK(Error, payload <= SIZE_MAX - slack);
    const size_t level_bytes = payload + slack;
    CHECK(Error, level_bytes <= SIZE_MAX - bytes);
    bytes += level_bytes;
  }

  *out_bytes = bytes;
  *out_run_count = runs;
  return 0;

Error:
  if (out_bytes)
    *out_bytes = 0;
  if (out_run_count)
    *out_run_count = 0;
  return 1;
}

int
host_batch_build_compact(struct host_batch* host,
                         void* aggregate_data,
                         size_t aggregate_capacity,
                         const size_t* offsets,
                         const size_t* chunk_sizes,
                         const struct batch_aggregate_layout* batch_layout,
                         const struct aggregate_layout* per_lod_layouts,
                         struct shard_state* const* shards_by_lod,
                         const uint32_t* per_lod_n_active,
                         uint8_t nlod,
                         enum host_delivery_policy policy,
                         size_t shard_alignment,
                         struct d2h_transfer_span* spans,
                         size_t span_capacity,
                         size_t* out_span_count,
                         void* slot_lifetime)
{
  CHECK(Error, host && aggregate_data && offsets && chunk_sizes);
  CHECK(Error, batch_layout && per_lod_layouts && shards_by_lod);
  CHECK(Error, per_lod_n_active && out_span_count);
  CHECK(Error, nlod == batch_layout->nlod);
  CHECK(Error,
        shard_alignment == 0 ? policy == HOST_DELIVERY_FIXED_TAIL ||
                                 policy == HOST_DELIVERY_INDEXED_COMPACT
                             : policy == HOST_DELIVERY_FIXED_TAIL ||
                                 policy == HOST_DELIVERY_INDEXED_PADDED);
  CHECK(Error, batch_layout->total_batch_covering <= SIZE_MAX - nlod);
  const size_t metadata_entries =
    (size_t)batch_layout->total_batch_covering + nlod;

  size_t run_count = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct shard_state* ss = shards_by_lod[lv];
    CHECK(Error, ss && ss->chunks_per_shard_append > 0);
    CHECK(Error, ss->epoch_in_shard < ss->chunks_per_shard_append);
    uint32_t left = per_lod_n_active[lv];
    uint64_t epoch = ss->epoch_in_shard;
    while (left > 0) {
      const uint64_t remaining = ss->chunks_per_shard_append - epoch;
      const uint32_t run = left < remaining ? left : (uint32_t)remaining;
      CHECK(Error, run > 0);
      CHECK(Error, ss->shard_inner_count <= SIZE_MAX - run_count);
      run_count += (size_t)ss->shard_inner_count;
      left -= run;
      epoch = run == remaining ? 0 : epoch + run;
    }
  }
  CHECK(Error, host_batch_reserve(host, run_count) == 0);
  CHECK(Error, run_count == 0 || spans);
  CHECK(Error, span_capacity >= run_count);

  host->run_count = 0;
  host->nlod = nlod;
  host->policy = policy;
  host->shard_alignment = shard_alignment;
  host->slot_lifetime = slot_lifetime;
  host->transfer.logical_payload_bytes = 0;
  *out_span_count = 0;
  size_t host_cursor = 0;
  const uintptr_t host_base = (uintptr_t)aggregate_data;

  for (uint8_t lv = 0; lv < nlod; ++lv) {
    struct shard_state* ss = shards_by_lod[lv];
    const struct aggregate_layout* al = &per_lod_layouts[lv];
    const struct lod_segment* seg = &batch_layout->lods[lv];
    const uint32_t n_active = per_lod_n_active[lv];
    const uint64_t cps = ss->chunks_per_shard_inner;
    CHECK(Error,
          seg->batch_covering_offset <= batch_layout->total_batch_covering);
    CHECK(Error, seg->batch_covering_offset <= SIZE_MAX - lv);
    const size_t metadata_base = seg->batch_covering_offset + (size_t)lv;
    const size_t page_size = shard_alignment;
    CHECK(Error, al->page_size == shard_alignment);
    CHECK(Error, al->page_size == per_lod_layouts[0].page_size);
    CHECK(Error, cps == seg->chunks_per_shard_inner);
    CHECK(Error, cps == al->cps_inner);
    CHECK(Error, ss->shard_inner_count == al->num_shards);
    CHECK(Error, ss->chunks_per_shard_append == al->chunks_per_shard_append);
    CHECK(Error, seg->n_active == n_active);
    CHECK(Error, seg->covering_count == al->covering_count);
    CHECK(Error, seg->chunks_per_epoch == al->chunks_per_epoch);
    CHECK_MUL_OVERFLOW(Error, n_active, al->covering_count, SIZE_MAX);
    const size_t segment_entries = (size_t)n_active * al->covering_count;
    CHECK(Error, metadata_base < metadata_entries);
    CHECK(Error, segment_entries <= metadata_entries - metadata_base - 1);
    CHECK(Error, seg->data_segment_offset <= batch_layout->total_data_bytes);
    CHECK(Error,
          seg->data_segment_bytes <=
            batch_layout->total_data_bytes - seg->data_segment_offset);

    uint32_t a = 0;
    uint64_t epoch = ss->epoch_in_shard;
    uint64_t generation = ss->shard_epoch;
    while (a < n_active) {
      const uint64_t remaining = ss->chunks_per_shard_append - epoch;
      const uint32_t left = n_active - a;
      const uint32_t run_len = left < remaining ? left : (uint32_t)remaining;
      const int finalizes = run_len == remaining;

      for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
        const uint64_t j = si * (uint64_t)n_active * cps + (uint64_t)a * cps;
        const uint64_t nchunks = (uint64_t)run_len * cps;
        CHECK(Error, j <= segment_entries);
        CHECK(Error, nchunks <= segment_entries - (size_t)j);
        size_t payload = 0;
        CHECK(Error,
              run_payload_bytes(
                chunk_sizes + metadata_base + j, nchunks, &payload) == 0);
        const size_t source = offsets[metadata_base + j];
        // Indexed compact aggregates pack actual bytes across LOD boundaries;
        // a later LOD can therefore begin before its worst-case capacity
        // offset in the static layout.
        CHECK(Error, source <= batch_layout->total_data_bytes);
        CHECK(Error, payload <= batch_layout->total_data_bytes - source);

        size_t region = host_cursor;
        size_t reserve = payload;
        const size_t tail = policy == HOST_DELIVERY_FIXED_TAIL && a == 0
                              ? ss->shards[si].tail_bytes
                              : 0;
        CHECK(Error, tail <= SIZE_MAX - payload);
        const size_t logical_run = tail + payload;
        const int needs_region = logical_run > 0;
        if (page_size > 0 && needs_region) {
          CHECK(Error, policy != HOST_DELIVERY_INDEXED_COMPACT);
          CHECK(Error, tail < page_size);
          const size_t address_remainder = (host_base + region) % page_size;
          if (address_remainder != 0) {
            const size_t pad = page_size - address_remainder;
            CHECK(Error, region <= SIZE_MAX - pad);
            region += pad;
          }
          if (policy == HOST_DELIVERY_FIXED_TAIL) {
            CHECK(Error, logical_run >= payload);
            reserve = logical_run;
          } else {
            CHECK(Error, policy == HOST_DELIVERY_INDEXED_PADDED);
            CHECK(Error,
                  checked_align_up_size(payload, page_size, &reserve) == 0);
          }
        } else if (!needs_region) {
          reserve = 0;
        } else {
          CHECK(Error,
                policy == HOST_DELIVERY_FIXED_TAIL ||
                  policy == HOST_DELIVERY_INDEXED_COMPACT);
          CHECK(Error, tail == 0);
        }
        CHECK(Error, region <= aggregate_capacity);
        CHECK(Error, reserve <= aggregate_capacity - region);
        CHECK(Error, tail <= reserve);
        CHECK(Error, payload <= reserve - tail);
        if (tail > 0) {
          CHECK(Error, ss->shards[si].tail_buf);
          memcpy(
            (uint8_t*)aggregate_data + region, ss->shards[si].tail_buf, tail);
        }
        if (policy == HOST_DELIVERY_INDEXED_PADDED && reserve > payload)
          memset(
            (uint8_t*)aggregate_data + region + payload, 0, reserve - payload);

        if (payload > 0) {
          spans[*out_span_count] = (struct d2h_transfer_span){
            .device_offset = source,
            .host_offset = region + tail,
            .bytes = payload,
          };
          (*out_span_count)++;
        }

        CHECK(Error, generation <= (UINT64_MAX - si) / ss->shard_inner_count);

        host->runs[host->run_count++] = (struct host_batch_run){
          .level = lv,
          .inner_shard = si,
          .flat_shard = generation * ss->shard_inner_count + si,
          .active_begin = a,
          .active_count = run_len,
          .epoch_in_shard = epoch,
          .chunks_per_shard_inner = cps,
          .finalizes = finalizes,
          .ends_generation_run = si + 1 == ss->shard_inner_count,
          .data = (uint8_t*)aggregate_data + region,
          .page_size = page_size,
          .tail_bytes = tail,
          .payload_bytes = payload,
          .source_offset = source,
          .offsets = offsets + metadata_base + j,
          .chunk_sizes = chunk_sizes + metadata_base + j,
        };
        CHECK(Error,
              payload <= SIZE_MAX - host->transfer.logical_payload_bytes);
        host->transfer.logical_payload_bytes += payload;
        CHECK(Error, region <= SIZE_MAX - reserve);
        host_cursor = region + reserve;
      }

      a += run_len;
      if (finalizes) {
        epoch = 0;
        CHECK(Error, generation < UINT64_MAX);
        generation++;
      } else {
        epoch += run_len;
      }
    }
  }

  CHECK(Error, host->run_count == run_count);
  return 0;

Error:
  if (host) {
    host->run_count = 0;
    host->nlod = 0;
    host->policy = HOST_DELIVERY_FIXED_TAIL;
    host->shard_alignment = 0;
  }
  if (out_span_count)
    *out_span_count = 0;
  return 1;
}

int
host_batch_build_legacy(struct host_batch* host,
                        void* aggregate_data,
                        const size_t* offsets,
                        const size_t* chunk_sizes,
                        const struct batch_aggregate_layout* batch_layout,
                        const struct aggregate_layout* per_lod_layouts,
                        struct shard_state* const* shards_by_lod,
                        const uint32_t* per_lod_n_active,
                        uint8_t nlod,
                        void* slot_lifetime)
{
  size_t* consumed = NULL;
  CHECK(Error, host && aggregate_data && offsets && chunk_sizes);
  CHECK(Error, batch_layout && per_lod_layouts && shards_by_lod);
  CHECK(Error, per_lod_n_active && nlod == batch_layout->nlod);

  size_t count = 0;
  for (uint8_t lv = 0; lv < nlod; ++lv) {
    const struct shard_state* ss = shards_by_lod[lv];
    CHECK(Error, ss);
    CHECK(Error, ss->chunks_per_shard_append > 0);
    CHECK(Error, ss->epoch_in_shard < ss->chunks_per_shard_append);
    uint32_t left = per_lod_n_active[lv];
    uint64_t epoch = ss->epoch_in_shard;
    while (left > 0) {
      uint64_t remain = ss->chunks_per_shard_append - epoch;
      uint32_t run = left < remain ? left : (uint32_t)remain;
      CHECK(Error, run > 0);
      CHECK(Error, ss->shard_inner_count <= SIZE_MAX - count);
      count += (size_t)ss->shard_inner_count;
      left -= run;
      epoch = run == remain ? 0 : epoch + run;
    }
  }
  CHECK(Error, host_batch_reserve(host, count) == 0);

  host->run_count = 0;
  host->nlod = nlod;
  host->policy = HOST_DELIVERY_FIXED_TAIL;
  host->shard_alignment = per_lod_layouts[0].page_size;
  host->slot_lifetime = slot_lifetime;
  host->transfer.logical_payload_bytes = 0;

  for (uint8_t lv = 0; lv < nlod; ++lv) {
    struct shard_state* ss = shards_by_lod[lv];
    const struct aggregate_layout* al = &per_lod_layouts[lv];
    const struct lod_segment* seg = &batch_layout->lods[lv];
    const uint32_t n_active = per_lod_n_active[lv];
    const uint64_t cps = ss->chunks_per_shard_inner;
    const size_t meta_base = seg->batch_covering_offset + (size_t)lv;
    uint32_t a = 0;
    uint64_t epoch = ss->epoch_in_shard;
    uint64_t generation = ss->shard_epoch;
    if (ss->shard_inner_count > 0) {
      consumed = (size_t*)calloc((size_t)ss->shard_inner_count, sizeof(size_t));
      CHECK(Error, consumed);
    }

    while (a < n_active) {
      const uint64_t remaining = ss->chunks_per_shard_append - epoch;
      const uint32_t left = n_active - a;
      const uint32_t run_len = left < remaining ? left : (uint32_t)remaining;
      const int finalizes = run_len == remaining;

      for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
        const uint64_t j = si * (uint64_t)n_active * cps + (uint64_t)a * cps;
        const uint64_t nchunks = (uint64_t)run_len * cps;
        size_t payload = 0;
        CHECK(Error,
              run_payload_bytes(
                chunk_sizes + meta_base + j, nchunks, &payload) == 0);
        const size_t tail = (a == 0) ? ss->shards[si].tail_bytes : 0;
        const size_t source = offsets[meta_base + j];
        CHECK(Error, source >= tail);

        struct host_batch_run* out = &host->runs[host->run_count++];
        *out = (struct host_batch_run){
          .level = lv,
          .inner_shard = si,
          .flat_shard = generation * ss->shard_inner_count + si,
          .active_begin = a,
          .active_count = run_len,
          .epoch_in_shard = epoch,
          .chunks_per_shard_inner = cps,
          .finalizes = finalizes,
          .ends_generation_run = si + 1 == ss->shard_inner_count,
          .data = (uint8_t*)aggregate_data + source - tail,
          .page_size = al->page_size,
          .tail_bytes = tail,
          .payload_bytes = payload,
          .source_offset = source,
          .offsets = offsets + meta_base + j,
          .chunk_sizes = chunk_sizes + meta_base + j,
        };

        // The legacy carry layout packs successive generation runs in the
        // same shard reservation.  This check catches a stale or malformed
        // offset table without imposing that layout on the host-tail design.
        if (al->page_size > 0 && al->shard_capacity > 0) {
          const size_t expected = seg->data_segment_offset +
                                  (size_t)si * al->shard_capacity +
                                  consumed[si] + tail;
          CHECK(Error, source == expected);
          consumed[si] += tail + payload;
        }

        CHECK(Error,
              payload <= SIZE_MAX - host->transfer.logical_payload_bytes);
        host->transfer.logical_payload_bytes += payload;
      }

      a += run_len;
      if (finalizes) {
        epoch = 0;
        generation++;
      } else {
        epoch += run_len;
      }
    }
    free(consumed);
    consumed = NULL;
  }

  CHECK(Error, host->run_count == count);
  return 0;

Error:
  free(consumed);
  host->run_count = 0;
  host->nlod = 0;
  return 1;
}

void
host_batch_destroy(struct host_batch* host)
{
  if (!host)
    return;
  free(host->runs);
  *host = (struct host_batch){ 0 };
}
