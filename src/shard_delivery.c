#include "shard_delivery.h"

#include "crc32c.h"
#include "platform.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

int
emit_shards(struct shard_state* ss, size_t shard_alignment)
{
  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    size_t index_data_bytes = ss->chunks_per_shard_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;

    uint8_t* index_buf;
    size_t write_bytes;
    size_t index_offset; // offset of index data within write buffer

    if (shard_alignment > 0) {
      // Unbuffered IO: pad BEFORE the index so the file ends sector-aligned
      // with the index at the very end. No truncation needed.
      write_bytes = align_up(index_total_bytes, shard_alignment);
      index_buf =
        (uint8_t*)platform_aligned_alloc(shard_alignment, write_bytes);
      CHECK(Error, index_buf);
      index_offset = write_bytes - index_total_bytes;
      memset(index_buf, 0, index_offset); // zero the padding before index
    } else {
      write_bytes = index_total_bytes;
      index_buf = (uint8_t*)malloc(write_bytes);
      CHECK(Error, index_buf);
      index_offset = 0;
    }

    memcpy(index_buf + index_offset, sh->index, index_data_bytes);

    uint32_t crc_val = crc32c(index_buf + index_offset, index_data_bytes);
    memcpy(index_buf + index_offset + index_data_bytes, &crc_val, 4);

    int wrc = sh->writer->write(
      sh->writer, sh->data_cursor, index_buf, index_buf + write_bytes);

    if (shard_alignment > 0)
      platform_aligned_free(index_buf);
    else
      free(index_buf);
    CHECK(Error, wrc == 0);

    CHECK(Error, sh->writer->finalize(sh->writer) == 0);

    sh->writer = NULL;
    sh->data_cursor = 0;
    memset(sh->index, 0xFF, ss->chunks_per_shard_total * 2 * sizeof(uint64_t));
  }

  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return 0;

Error:
  return 1;
}

int
deliver_to_shards_batch(uint8_t level,
                        struct shard_state* ss,
                        struct aggregate_result* result,
                        uint32_t n_active,
                        struct shard_sink* sink,
                        size_t shard_alignment,
                        size_t* out_bytes)
{
  const uint64_t cps_inner = ss->chunks_per_shard_inner;
  const size_t sa = shard_alignment;
  size_t total_bytes = 0;

  // Process epochs in runs: a run is a contiguous sequence of epochs that
  // belong to the same shard (no shard completion boundary in between).
  // Writing all epochs in a run with one write_direct call reduces syscalls.
  uint32_t a = 0;
  while (a < n_active) {
    uint32_t remaining_in_shard =
      (uint32_t)(ss->chunks_per_shard_0 - ss->epoch_in_shard);
    uint32_t remaining_in_batch = n_active - a;
    uint32_t run_len = remaining_in_shard < remaining_in_batch
                         ? remaining_in_shard
                         : remaining_in_batch;

    for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
      struct active_shard* sh = &ss->shards[si];

      if (!sh->writer) {
        uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
        sh->writer = sink->open(sink, level, flat);
        CHECK(Error, sh->writer);
      }

      // Contiguous range in aggregated buffer for this run
      uint64_t j_run_start = si * n_active * cps_inner + a * cps_inner;
      uint64_t j_run_end = j_run_start + (uint64_t)run_len * cps_inner;

      size_t run_bytes =
        result->offsets[j_run_end] - result->offsets[j_run_start];
      if (run_bytes > 0) {
        const void* src =
          (const char*)result->data + result->offsets[j_run_start];
        // Unbuffered IO: round write size up to alignment. The padding
        // region in h_aggregated is safe to read (buffer is oversized).
        size_t write_bytes = sa > 0 ? align_up(run_bytes, sa) : run_bytes;
        total_bytes += write_bytes;
        const void* src_end = (const char*)src + write_bytes;

        // Use write_direct when source pointer is page-aligned (always true
        // at shard-group boundaries; may not hold after a mid-batch shard
        // completion splits a group).
        // FIXME: this logic should be handle by a wrapper that is exposed in
        //        writer.h - something like write_append()
        int aligned = sa == 0 || ((uintptr_t)src % sa == 0);
        if (aligned && sh->writer->write_direct) {
          CHECK(Error,
                sh->writer->write_direct(
                  sh->writer, sh->data_cursor, src, src_end) == 0);
        } else {
          CHECK(Error,
                sh->writer->write(sh->writer, sh->data_cursor, src, src_end) ==
                  0);
        }
      }

      // Record shard index entries for each epoch in the run
      for (uint32_t r = 0; r < run_len; ++r) {
        uint64_t eis = ss->epoch_in_shard + r;
        uint64_t j_start = j_run_start + (uint64_t)r * cps_inner;
        for (uint64_t j = j_start; j < j_start + cps_inner; ++j) {
          size_t chunk_size = result->chunk_sizes[j];
          if (chunk_size > 0) {
            uint64_t within_inner = j - j_start;
            uint64_t slot_idx = eis * cps_inner + within_inner;
            size_t chunk_off = sh->data_cursor + (result->offsets[j] -
                                                  result->offsets[j_run_start]);
            sh->index[2 * slot_idx] = chunk_off;
            sh->index[2 * slot_idx + 1] = chunk_size;
          }
        }
      }

      sh->data_cursor += sa > 0 ? align_up(run_bytes, sa) : run_bytes;
    }

    ss->epoch_in_shard += run_len;
    a += run_len;

    if (ss->epoch_in_shard >= ss->chunks_per_shard_0) {
      CHECK(Error, emit_shards(ss, sa) == 0);
    }
  }

  if (out_bytes)
    *out_bytes = total_bytes;
  return 0;

Error:
  return 1;
}
