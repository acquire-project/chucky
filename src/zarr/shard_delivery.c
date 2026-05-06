#include "zarr/shard_delivery.h"

#include "platform/platform.h"
#include "util/prelude.h"
#include "zarr/crc32c.h"

#include "log/log.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

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
  for (uint64_t si = 0; si < li->shard_inner_count; ++si) {
    ss->shards[si].index =
      (uint64_t*)malloc(li->chunks_per_shard_total * 2 * sizeof(uint64_t));
    if (!ss->shards[si].index)
      return 1;
    memset(ss->shards[si].index,
           0xFF,
           li->chunks_per_shard_total * 2 * sizeof(uint64_t));
    if (page > 0) {
      ss->shards[si].tail_buf = (uint8_t*)malloc(page);
      if (!ss->shards[si].tail_buf)
        return 1;
    }
  }
  return 0;
}

// Build the page-aligned scratch holding [tail][index][crc4][zero pad].
// Caller frees via free_finalize_buf with the same shard_alignment.
static int
build_finalize_buf(const uint8_t* tail_src,
                   size_t tail_bytes,
                   const uint64_t* index_data,
                   size_t index_data_bytes,
                   size_t shard_alignment,
                   uint8_t** out_buf,
                   size_t* out_aligned_bytes,
                   size_t* out_logical_bytes)
{
  size_t logical_bytes = tail_bytes + index_data_bytes + 4;
  size_t aligned_bytes = shard_alignment > 0
                           ? align_up(logical_bytes, shard_alignment)
                           : logical_bytes;
  uint8_t* buf =
    shard_alignment > 0
      ? (uint8_t*)platform_aligned_alloc(shard_alignment, aligned_bytes)
      : (uint8_t*)malloc(aligned_bytes);
  if (!buf)
    return 1;

  if (tail_bytes > 0)
    memcpy(buf, tail_src, tail_bytes);
  memcpy(buf + tail_bytes, index_data, index_data_bytes);
  uint32_t crc_val = crc32c(buf + tail_bytes, index_data_bytes);
  memcpy(buf + tail_bytes + index_data_bytes, &crc_val, 4);
  if (aligned_bytes > logical_bytes)
    memset(buf + logical_bytes, 0, aligned_bytes - logical_bytes);

  *out_buf = buf;
  *out_aligned_bytes = aligned_bytes;
  *out_logical_bytes = logical_bytes;
  return 0;
}

static void
free_finalize_buf(uint8_t* buf, size_t shard_alignment)
{
  if (shard_alignment > 0)
    platform_aligned_free(buf);
  else
    free(buf);
}

// Synchronize the per-shard tail length and bytes across the active_shard and
// the host shadow array. n == 0 clears (no copy). Use this everywhere the tail
// state changes so the two sources of truth never drift.
static void
shard_tail_set(struct active_shard* sh,
               size_t* h_tail_bytes_si,
               const uint8_t* src,
               size_t n)
{
  sh->tail_bytes = n;
  *h_tail_bytes_si = n;
  if (n > 0)
    memcpy(sh->tail_buf, src, n);
}

int
finalize_shards(struct shard_state* ss, size_t shard_alignment)
{
  int err = 0;
  size_t index_data_bytes = ss->chunks_per_shard_total * 2 * sizeof(uint64_t);

  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    uint8_t* buf = NULL;
    size_t aligned_bytes = 0;
    size_t logical_bytes = 0;

    if (build_finalize_buf(sh->tail_buf,
                           sh->tail_bytes,
                           sh->index,
                           index_data_bytes,
                           shard_alignment,
                           &buf,
                           &aligned_bytes,
                           &logical_bytes)) {
      log_error("finalize_shards: alloc failed for shard %llu",
                (unsigned long long)si);
      err = 1;
    } else {
      if (sh->writer->write(
            sh->writer, sh->data_cursor, buf, buf + aligned_bytes)) {
        log_error("finalize_shards: write failed for shard %llu",
                  (unsigned long long)si);
        err = 1;
      }
      free_finalize_buf(buf, shard_alignment);
    }

    // Trim the trailing pad so shard_index_parse finds the index at the file's
    // end. No-op for sinks whose truncate hook is NULL (e.g. S3).
    if (!err && sh->writer->truncate) {
      uint64_t logical_size = sh->data_cursor + (uint64_t)logical_bytes;
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

  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return err;
}

// Sum chunk_sizes for chunks in shard si over epoch range [a, a+run_len).
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

// Record the [eis_start, eis_start+run_len) shard-index entries for the run.
// chunk_off is computed as sh->data_cursor + (offset_in_agg - base_off), where
// base_off is the agg-buffer offset of the run's first chunk.
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

// Carry-over finalizing run: bundle [tail || data || index || crc] into a
// single page-aligned write, then truncate the file to the logical end. Must
// use the copying write since fbuf is freed before the worker drains.
static int
deliver_run_finalizing(struct active_shard* sh,
                       const struct shard_state* ss,
                       const uint8_t* src,
                       size_t total_run,
                       size_t* h_tail_bytes_si,
                       size_t sa,
                       size_t* total_bytes)
{
  size_t index_data_bytes = ss->chunks_per_shard_total * 2 * sizeof(uint64_t);
  uint8_t* fbuf = NULL;
  size_t aligned_bytes = 0;
  size_t logical_bytes = 0;
  if (build_finalize_buf(src,
                         total_run,
                         sh->index,
                         index_data_bytes,
                         sa,
                         &fbuf,
                         &aligned_bytes,
                         &logical_bytes))
    return 1;

  int wr =
    sh->writer->write(sh->writer, sh->data_cursor, fbuf, fbuf + aligned_bytes);
  free_finalize_buf(fbuf, sa);
  if (wr)
    return 1;
  *total_bytes += aligned_bytes;

  if (sh->writer->truncate) {
    uint64_t logical_size = (uint64_t)sh->data_cursor + (uint64_t)logical_bytes;
    if (sh->writer->truncate(sh->writer, logical_size))
      return 1;
  }

  shard_tail_set(sh, h_tail_bytes_si, NULL, 0);
  return 0;
}

// Carry-over non-finalizing run: write the page-aligned floor; capture the
// sub-page remainder as the next batch's leading tail. write_direct when src
// is page-aligned, else copy.
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

// Non-carry-over path: agg buffer is a single contiguous prefix-sum across
// all shards (no per-shard regions, no leading-tail). One write per run, then
// index recording, then cursor advance. Used whenever the sink has no
// alignment requirement (buffered FS, S3, in-memory).
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

// Carry-over post-generation cleanup: the bundled finalize write already
// emitted the index. Just call writer->finalize and reset per-shard state.
static int
close_shards_after_bundle(struct shard_state* ss)
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
                        size_t* out_bytes)
{
  const uint64_t cps_inner = ss->chunks_per_shard_inner;
  const size_t sa = shard_alignment;
  const size_t page_size = layout ? layout->page_size : 0;
  const size_t shard_capacity = layout ? layout->shard_capacity : 0;
  const int use_carryover = (page_size > 0 && shard_capacity > 0);
  // Whether the agg buffer pads gen boundaries — bytes_consumed must skip the
  // pad after a finalizing run so the next gen's src is page-aligned.
  const int requires_gen_pads = (layout && layout->requires_gen_pads);
  size_t total_bytes = 0;

  // Precondition: tail-carry callers must supply h_tail_bytes. Legacy callers
  // (page_size == 0) may pass NULL.
  assert(!use_carryover || h_tail_bytes != NULL);

  // Per-shard cumulative bytes consumed from the agg buffer in this batch.
  // Advances per run for each shard so subsequent runs (after intra-batch
  // shard finalizes) read from the correct offset.
  size_t* bytes_consumed = NULL;
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
      }

      uint64_t j_run_start = si * n_active * cps_inner + a * cps_inner;
      uint64_t j_run_end = j_run_start + (uint64_t)run_len * cps_inner;

      if (use_carryover) {
        // Each shard owns a fixed page-aligned region of `shard_capacity`
        // bytes. The first run for a shard sees the leading tail (carried
        // from the prior batch); subsequent runs in the same batch start
        // past the consumed bytes. The aggregator may have padded gen
        // boundaries (requires_gen_pads); deliver tracks that pad in
        // bytes_consumed so the next gen's src lands page-aligned.
        const size_t shard_base = (size_t)si * shard_capacity;
        const int is_first_run_for_shard = (bytes_consumed[si] == 0);
        const size_t tail_in = is_first_run_for_shard ? h_tail_bytes[si] : 0;
        const size_t run_real = sum_run_chunks(
          result->chunk_sizes, si, a, run_len, n_active, cps_inner);
        const size_t total_run = tail_in + run_real;
        const size_t src_offset_in_shard = bytes_consumed[si];
        const uint8_t* src =
          (const uint8_t*)result->data + shard_base + src_offset_in_shard;

        // Record index entries before any write so the finalize bundle's
        // baked-in index is up to date.
        record_run_index(sh,
                         result,
                         j_run_start,
                         shard_base + src_offset_in_shard,
                         run_len,
                         ss->epoch_in_shard,
                         cps_inner);

        if (run_finalizes) {
          CHECK(
            Error,
            deliver_run_finalizing(
              sh, ss, src, total_run, &h_tail_bytes[si], sa, &total_bytes) ==
              0);
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
        if (run_finalizes && requires_gen_pads)
          bytes_consumed[si] = align_up(bytes_consumed[si], page_size);
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
        CHECK(Error, close_shards_after_bundle(ss) == 0);
      else
        CHECK(Error, finalize_shards(ss, sa) == 0);
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
