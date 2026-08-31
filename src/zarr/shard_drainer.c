#include "zarr/shard_drainer.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/metric.h"
#include "zarr/crc32c.h"

#include <stdlib.h>
#include <string.h>

static int
checked_add_size(size_t a, size_t b, size_t* out)
{
  if (!out || a > SIZE_MAX - b)
    return 1;
  *out = a + b;
  return 0;
}

static int
checked_align_size(size_t value, size_t alignment, size_t* out)
{
  if (!out || alignment == 0)
    return 1;
  const size_t rem = value % alignment;
  const size_t add = rem == 0 ? 0 : alignment - rem;
  return checked_add_size(value, add, out);
}

static int
checked_index_bytes(const struct shard_state* ss, size_t* out)
{
  if (!ss || !out ||
      ss->chunks_per_shard_total > SIZE_MAX / (2 * sizeof(uint64_t)))
    return 1;
  *out = (size_t)ss->chunks_per_shard_total * 2 * sizeof(uint64_t);
  return 0;
}

static int
drain_context(struct shard_drainer* drain,
              const struct host_batch_run** out_run,
              struct shard_state** out_ss,
              struct active_shard** out_sh)
{
  if (!drain || !drain->host || !drain->shards_by_lod ||
      drain->run_index >= drain->host->run_count)
    return 1;
  const struct host_batch_run* run = &drain->host->runs[drain->run_index];
  if (run->level >= drain->host->nlod || !run->data)
    return 1;
  struct shard_state* ss = drain->shards_by_lod[run->level];
  if (!ss || !ss->shards || run->inner_shard >= ss->shard_inner_count ||
      ss->shard_inner_count == 0 || run->active_count == 0 ||
      ss->chunks_per_shard_inner == 0 || ss->chunks_per_shard_append == 0 ||
      ss->epoch_in_shard != run->epoch_in_shard ||
      run->chunks_per_shard_inner != ss->chunks_per_shard_inner ||
      run->epoch_in_shard >= ss->chunks_per_shard_append ||
      run->active_count > ss->chunks_per_shard_append - run->epoch_in_shard)
    return 1;
  if (ss->shard_epoch > (UINT64_MAX - run->inner_shard) / ss->shard_inner_count)
    return 1;
  const uint64_t flat =
    ss->shard_epoch * ss->shard_inner_count + run->inner_shard;
  if (flat != run->flat_shard)
    return 1;
  if (!!run->ends_generation_run !=
      (run->inner_shard + 1 == ss->shard_inner_count))
    return 1;
  if (!!run->finalizes !=
      (run->active_count == ss->chunks_per_shard_append - run->epoch_in_shard))
    return 1;
  if (!ss->shards[run->inner_shard].index)
    return 1;
  *out_run = run;
  *out_ss = ss;
  *out_sh = &ss->shards[run->inner_shard];
  return 0;
}

static int
validate_run_index(const struct host_batch_run* run,
                   const struct shard_state* ss,
                   uint64_t payload_file_offset)
{
  if (!run || !ss || !run->offsets || !run->chunk_sizes)
    return 1;
  if (run->active_count > UINT64_MAX / run->chunks_per_shard_inner)
    return 1;
  const uint64_t nchunks =
    (uint64_t)run->active_count * run->chunks_per_shard_inner;
  size_t expected_relative = 0;
  for (uint64_t j = 0; j < nchunks; ++j) {
    const size_t n = run->chunk_sizes[j];
    if (run->offsets[j] < run->source_offset)
      return 1;
    const size_t relative = run->offsets[j] - run->source_offset;
    if (relative != expected_relative || relative > run->payload_bytes ||
        n > run->payload_bytes - relative ||
        payload_file_offset > UINT64_MAX - relative ||
        payload_file_offset + relative > UINT64_MAX - n)
      return 1;
    expected_relative += n;
    if (n == 0)
      continue;
    const uint64_t r = j / run->chunks_per_shard_inner;
    const uint64_t c = j % run->chunks_per_shard_inner;
    if (run->epoch_in_shard > UINT64_MAX - r)
      return 1;
    const uint64_t epoch = run->epoch_in_shard + r;
    if (epoch >= ss->chunks_per_shard_append ||
        epoch > UINT64_MAX / run->chunks_per_shard_inner)
      return 1;
    const uint64_t slot = epoch * run->chunks_per_shard_inner + c;
    if (slot >= ss->chunks_per_shard_total)
      return 1;
  }
  return expected_relative == run->payload_bytes ? 0 : 1;
}

static void
commit_run_index(struct active_shard* sh,
                 const struct host_batch_run* run,
                 uint64_t payload_file_offset)
{
  const uint64_t cps = run->chunks_per_shard_inner;
  for (uint32_t r = 0; r < run->active_count; ++r) {
    for (uint64_t c = 0; c < cps; ++c) {
      const uint64_t j = (uint64_t)r * cps + c;
      const size_t n = run->chunk_sizes[j];
      if (n == 0)
        continue;
      const uint64_t slot = (run->epoch_in_shard + r) * cps + c;
      sh->index[2 * slot] =
        payload_file_offset + (run->offsets[j] - run->source_offset);
      sh->index[2 * slot + 1] = n;
    }
  }
}

static void
patch_footer_index(uint8_t* index,
                   const struct host_batch_run* run,
                   uint64_t payload_file_offset)
{
  const uint64_t cps = run->chunks_per_shard_inner;
  for (uint32_t r = 0; r < run->active_count; ++r) {
    for (uint64_t c = 0; c < cps; ++c) {
      const uint64_t j = (uint64_t)r * cps + c;
      const size_t n = run->chunk_sizes[j];
      if (n == 0)
        continue;
      const uint64_t slot = (run->epoch_in_shard + r) * cps + c;
      const uint64_t offset =
        payload_file_offset + (run->offsets[j] - run->source_offset);
      const uint64_t size = n;
      memcpy(index + (2 * slot) * sizeof(uint64_t), &offset, sizeof(offset));
      memcpy(index + (2 * slot + 1) * sizeof(uint64_t), &size, sizeof(size));
    }
  }
}

static void
finish_run(struct shard_drainer* drain)
{
  drain->run_index++;
  drain->phase = SHARD_DRAIN_PHASE_RUN;
  drain->metric_recorded = 0;
  drain->data_physical_bytes = 0;
  drain->footer_remainder_bytes = 0;
  drain->footer_logical_bytes = 0;
  drain->footer_physical_bytes = 0;
}

static void
advance_nonfinal(struct shard_state* ss, const struct host_batch_run* run)
{
  if (run->ends_generation_run)
    ss->epoch_in_shard += run->active_count;
}

static int
start_run(struct shard_drainer* drain)
{
  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (drain_context(drain, &run, &ss, &sh))
    return 1;
  if (run->page_size != drain->shard_alignment)
    return 1;

  const enum host_delivery_policy policy = drain->host->policy;
  const size_t alignment = drain->shard_alignment;
  if ((policy == HOST_DELIVERY_INDEXED_PADDED && alignment == 0) ||
      (policy == HOST_DELIVERY_INDEXED_COMPACT && alignment != 0))
    return 1;
  if (policy != HOST_DELIVERY_FIXED_TAIL && sh->tail_bytes != 0)
    return 1;
  if (policy == HOST_DELIVERY_FIXED_TAIL && alignment == 0 &&
      run->tail_bytes != 0)
    return 1;

  size_t total = run->payload_bytes;
  if (policy == HOST_DELIVERY_FIXED_TAIL && alignment > 0) {
    if (run->tail_bytes != sh->tail_bytes ||
        checked_add_size(run->tail_bytes, run->payload_bytes, &total))
      return 1;
  } else if (run->tail_bytes != 0) {
    return 1;
  }

  drain->run_start_cursor = sh->data_cursor;
  drain->data_physical_bytes = run->payload_bytes;
  drain->footer_remainder_bytes = 0;
  if (alignment > 0 && policy != HOST_DELIVERY_INDEXED_COMPACT) {
    if (run->finalizes) {
      drain->data_physical_bytes = (total / alignment) * alignment;
      drain->footer_remainder_bytes = total - drain->data_physical_bytes;
    } else if (policy == HOST_DELIVERY_FIXED_TAIL) {
      drain->data_physical_bytes = (total / alignment) * alignment;
    } else if (checked_align_size(
                 run->payload_bytes, alignment, &drain->data_physical_bytes)) {
      return 1;
    }
  }

  if (sh->data_cursor > UINT64_MAX - drain->data_physical_bytes)
    return 1;
  uint64_t payload_file_offset = drain->run_start_cursor;
  if (policy == HOST_DELIVERY_FIXED_TAIL && alignment > 0) {
    if (payload_file_offset > UINT64_MAX - run->tail_bytes)
      return 1;
    payload_file_offset += run->tail_bytes;
  }
  if (validate_run_index(run, ss, payload_file_offset))
    return 1;
  if (!run->finalizes && policy == HOST_DELIVERY_FIXED_TAIL && alignment > 0) {
    const size_t remainder = total - drain->data_physical_bytes;
    if (remainder >= alignment || (remainder > 0 && !sh->tail_buf))
      return 1;
  }
  if (run->finalizes && run->ends_generation_run) {
    if (ss->shard_epoch == UINT64_MAX ||
        ss->shard_epoch + 1 > UINT64_MAX / ss->chunks_per_shard_append)
      return 1;
  }

  if (run->finalizes) {
    size_t index_bytes = 0;
    size_t logical = 0;
    if (checked_index_bytes(ss, &index_bytes) ||
        checked_add_size(
          drain->footer_remainder_bytes, index_bytes, &logical) ||
        checked_add_size(logical, 4, &logical))
      return 1;
    drain->footer_logical_bytes = logical;
    drain->footer_physical_bytes = logical;
    if (alignment > 0 &&
        checked_align_size(logical, alignment, &drain->footer_physical_bytes))
      return 1;
    if (sh->data_cursor + drain->data_physical_bytes >
        UINT64_MAX - drain->footer_logical_bytes)
      return 1;
    drain->phase = drain->data_physical_bytes > 0 ? SHARD_DRAIN_PHASE_DATA
                                                  : SHARD_DRAIN_PHASE_FOOTER;
    return 0;
  }

  // Empty indexed updates still yield a zero-byte DATA command. The executor
  // accepts it without opening a writer, which keeps every index/epoch
  // transition behind shard_drain_accept while emitting no padding or write.
  drain->phase = SHARD_DRAIN_PHASE_DATA;
  return 0;
}

static int
command_common(struct shard_drainer* drain,
               struct shard_drain_command* command,
               enum shard_drain_command_kind kind,
               const struct host_batch_run* run)
{
  if (drain->next_serial == UINT64_MAX)
    return 1;
  memset(command, 0, sizeof(*command));
  command->kind = kind;
  command->serial = ++drain->next_serial;
  command->level = run->level;
  command->inner_shard = run->inner_shard;
  command->flat_shard = run->flat_shard;
  return 0;
}

static int
command_matches(const struct shard_drain_command* a,
                const struct shard_drain_command* b)
{
  return a->kind == b->kind && a->serial == b->serial && a->level == b->level &&
         a->inner_shard == b->inner_shard && a->flat_shard == b->flat_shard &&
         a->file_offset == b->file_offset &&
         a->source_begin == b->source_begin && a->source_end == b->source_end &&
         a->physical_bytes == b->physical_bytes &&
         a->logical_size == b->logical_size &&
         a->direct_write_eligible == b->direct_write_eligible &&
         a->buffer_lease.kind == b->buffer_lease.kind &&
         a->buffer_lease.owner == b->buffer_lease.owner &&
         a->logical_payload_bytes == b->logical_payload_bytes &&
         a->internal_padding_bytes == b->internal_padding_bytes &&
         a->counts_physical_update == b->counts_physical_update &&
         a->counts_padded_update == b->counts_padded_update &&
         a->closes_generation == b->closes_generation;
}

int
shard_drain_begin(struct shard_drainer* drain,
                  struct host_batch* host,
                  struct shard_state* const* shards_by_lod)
{
  if (!drain || !host || !shards_by_lod ||
      (host->run_count > 0 && !host->runs) || host->nlod == 0 ||
      host->nlod > LOD_MAX_LEVELS)
    return 1;
  memset(drain, 0, sizeof(*drain));
  drain->host = host;
  drain->shards_by_lod = shards_by_lod;
  drain->shard_alignment = host->shard_alignment;
  drain->phase = SHARD_DRAIN_PHASE_RUN;
  if (host->shard_alignment == 0
        ? host->policy != HOST_DELIVERY_FIXED_TAIL &&
            host->policy != HOST_DELIVERY_INDEXED_COMPACT
        : host->policy != HOST_DELIVERY_FIXED_TAIL &&
            host->policy != HOST_DELIVERY_INDEXED_PADDED) {
    shard_drain_abort(drain);
    return 1;
  }
  return 0;
}

int
shard_drain_next(struct shard_drainer* drain,
                 struct shard_drain_command* command)
{
  if (!drain || !command || drain->failed || drain->pending)
    return -1;

  while (drain->phase == SHARD_DRAIN_PHASE_RUN) {
    if (drain->run_index == drain->host->run_count)
      return 0;
    if (start_run(drain)) {
      shard_drain_abort(drain);
      return -1;
    }
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (drain_context(drain, &run, &ss, &sh)) {
    shard_drain_abort(drain);
    return -1;
  }

  if (drain->phase == SHARD_DRAIN_PHASE_DATA) {
    if (command_common(drain, command, SHARD_DRAIN_DATA, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->source_begin = run->data;
    command->source_end = run->data + drain->data_physical_bytes;
    command->physical_bytes = drain->data_physical_bytes;
    command->buffer_lease = (struct shard_drain_buffer_lease){
      .kind = SHARD_DRAIN_LEASE_HOST_BATCH,
      .owner = drain->host->slot_lifetime,
    };
    if (command->physical_bytes > 0) {
      if (drain->shard_alignment == 0) {
        command->direct_write_eligible = 1;
      } else {
        const size_t a = drain->shard_alignment;
        command->direct_write_eligible =
          (uintptr_t)command->source_begin % a == 0 &&
          command->file_offset % a == 0 && command->physical_bytes % a == 0;
      }
    }
    if (!drain->metric_recorded && run->payload_bytes > 0) {
      command->logical_payload_bytes = run->payload_bytes;
      command->counts_physical_update = 1;
      if (!run->finalizes &&
          drain->host->policy == HOST_DELIVERY_INDEXED_PADDED) {
        command->internal_padding_bytes =
          drain->data_physical_bytes - run->payload_bytes;
        command->counts_padded_update = command->internal_padding_bytes > 0;
      }
    }
  } else if (drain->phase == SHARD_DRAIN_PHASE_FOOTER) {
    if (command_common(drain, command, SHARD_DRAIN_FOOTER, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->physical_bytes = drain->footer_physical_bytes;
    if (drain->shard_alignment > 0) {
      if (!sh->footer_buf ||
          drain->footer_physical_bytes > ss->footer_capacity) {
        shard_drain_abort(drain);
        return -1;
      }
      command->buffer_lease = (struct shard_drain_buffer_lease){
        .kind = SHARD_DRAIN_LEASE_FOOTER,
        .owner = sh,
      };
      command->source_begin = sh->footer_buf;
      command->source_end = sh->footer_buf + drain->footer_physical_bytes;
      const size_t a = drain->shard_alignment;
      command->direct_write_eligible = (uintptr_t)sh->footer_buf % a == 0 &&
                                       command->file_offset % a == 0 &&
                                       command->physical_bytes % a == 0;
    } else {
      command->buffer_lease = (struct shard_drain_buffer_lease){
        .kind = SHARD_DRAIN_LEASE_TRANSIENT,
        .owner = drain,
      };
    }
    if (!drain->metric_recorded && run->payload_bytes > 0) {
      command->logical_payload_bytes = run->payload_bytes;
      command->counts_physical_update = 1;
    }
  } else if (drain->phase == SHARD_DRAIN_PHASE_TRUNCATE) {
    if (command_common(drain, command, SHARD_DRAIN_TRUNCATE, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->logical_size = sh->data_cursor + drain->footer_logical_bytes;
  } else if (drain->phase == SHARD_DRAIN_PHASE_FINALIZE) {
    if (command_common(drain, command, SHARD_DRAIN_FINALIZE, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->closes_generation = run->ends_generation_run;
  } else {
    shard_drain_abort(drain);
    return -1;
  }

  drain->current = *command;
  drain->pending = 1;
  drain->prepared = command->kind != SHARD_DRAIN_FOOTER;
  if (command->kind == SHARD_DRAIN_FOOTER &&
      command->buffer_lease.kind == SHARD_DRAIN_LEASE_TRANSIENT &&
      shard_drain_prepare(drain, command))
    return -1;
  return 1;

Error:
  shard_drain_abort(drain);
  return -1;
}

int
shard_drain_prepare(struct shard_drainer* drain,
                    struct shard_drain_command* command)
{
  if (!drain || !command || drain->failed || !drain->pending ||
      command->serial != drain->current.serial ||
      command->kind != drain->current.kind)
    return 1;
  if (command->kind != SHARD_DRAIN_FOOTER) {
    *command = drain->current;
    return 0;
  }
  if (drain->prepared) {
    *command = drain->current;
    return 0;
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  size_t index_bytes = 0;
  if (drain_context(drain, &run, &ss, &sh) ||
      checked_index_bytes(ss, &index_bytes)) {
    shard_drain_abort(drain);
    return 1;
  }

  uint8_t* dst = sh->footer_buf;
  if (drain->shard_alignment == 0) {
    drain->transient_footer = (uint8_t*)malloc(drain->footer_physical_bytes);
    if (!drain->transient_footer) {
      shard_drain_abort(drain);
      return 1;
    }
    dst = drain->transient_footer;
  }
  if (!dst) {
    shard_drain_abort(drain);
    return 1;
  }

  if (drain->footer_remainder_bytes > 0)
    memcpy(dst,
           run->data + drain->data_physical_bytes,
           drain->footer_remainder_bytes);
  uint8_t* index_dst = dst + drain->footer_remainder_bytes;
  memcpy(index_dst, sh->index, index_bytes);
  uint64_t payload_file_offset = drain->run_start_cursor;
  if (drain->host->policy == HOST_DELIVERY_FIXED_TAIL &&
      drain->shard_alignment > 0)
    payload_file_offset += run->tail_bytes;
  patch_footer_index(index_dst, run, payload_file_offset);
  const uint32_t crc = crc32c(index_dst, index_bytes);
  memcpy(index_dst + index_bytes, &crc, sizeof(crc));
  if (drain->footer_physical_bytes > drain->footer_logical_bytes)
    memset(dst + drain->footer_logical_bytes,
           0,
           drain->footer_physical_bytes - drain->footer_logical_bytes);

  drain->current.source_begin = dst;
  drain->current.source_end = dst + drain->footer_physical_bytes;
  drain->prepared = 1;
  *command = drain->current;
  return 0;
}

int
shard_drain_accept(struct shard_drainer* drain,
                   const struct shard_drain_command* command)
{
  if (!drain || !command || drain->failed || !drain->pending ||
      !drain->prepared)
    return 1;
  if (!command_matches(command, &drain->current)) {
    shard_drain_abort(drain);
    return 1;
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (drain_context(drain, &run, &ss, &sh)) {
    shard_drain_abort(drain);
    return 1;
  }
  if (command->counts_physical_update)
    drain->metric_recorded = 1;

  if (command->kind == SHARD_DRAIN_DATA) {
    sh->data_cursor += drain->data_physical_bytes;
    if (run->finalizes) {
      if (drain->shard_alignment == 0) {
        commit_run_index(sh, run, drain->run_start_cursor);
      }
      drain->phase = SHARD_DRAIN_PHASE_FOOTER;
    } else {
      uint64_t payload_file_offset = drain->run_start_cursor;
      if (drain->host->policy == HOST_DELIVERY_FIXED_TAIL &&
          drain->shard_alignment > 0)
        payload_file_offset += run->tail_bytes;
      commit_run_index(sh, run, payload_file_offset);
      if (drain->host->policy == HOST_DELIVERY_FIXED_TAIL &&
          drain->shard_alignment > 0) {
        const size_t total = run->tail_bytes + run->payload_bytes;
        const size_t remainder = total - drain->data_physical_bytes;
        if (remainder > 0)
          memcpy(
            sh->tail_buf, run->data + drain->data_physical_bytes, remainder);
        sh->tail_bytes = remainder;
      } else {
        sh->tail_bytes = 0;
      }
      advance_nonfinal(ss, run);
      finish_run(drain);
    }
  } else if (command->kind == SHARD_DRAIN_FOOTER) {
    free(drain->transient_footer);
    drain->transient_footer = NULL;
    if (drain->shard_alignment > 0) {
      uint64_t payload_file_offset = drain->run_start_cursor;
      if (drain->host->policy == HOST_DELIVERY_FIXED_TAIL)
        payload_file_offset += run->tail_bytes;
      commit_run_index(sh, run, payload_file_offset);
    }
    sh->tail_bytes = 0;
    drain->phase = SHARD_DRAIN_PHASE_TRUNCATE;
  } else if (command->kind == SHARD_DRAIN_TRUNCATE) {
    drain->phase = SHARD_DRAIN_PHASE_FINALIZE;
  } else if (command->kind == SHARD_DRAIN_FINALIZE) {
    size_t index_bytes = 0;
    if (checked_index_bytes(ss, &index_bytes)) {
      shard_drain_abort(drain);
      return 1;
    }
    sh->writer = NULL;
    sh->data_cursor = 0;
    sh->tail_bytes = 0;
    memset(sh->index, 0xFF, index_bytes);
    if (run->ends_generation_run) {
      ss->finalized_append_chunks =
        (ss->shard_epoch + 1) * ss->chunks_per_shard_append;
      ss->epoch_in_shard = 0;
      ss->shard_epoch++;
    }
    finish_run(drain);
  } else {
    shard_drain_abort(drain);
    return 1;
  }

  drain->pending = 0;
  drain->prepared = 0;
  memset(&drain->current, 0, sizeof(drain->current));
  return 0;
}

void
shard_drain_abort(struct shard_drainer* drain)
{
  if (!drain)
    return;
  free(drain->transient_footer);
  drain->transient_footer = NULL;
  drain->pending = 0;
  drain->prepared = 0;
  drain->failed = 1;
}

void
shard_drain_destroy(struct shard_drainer* drain)
{
  if (!drain)
    return;
  free(drain->transient_footer);
  memset(drain, 0, sizeof(*drain));
}

static void
wait_footer_fence(struct shard_sink* sink,
                  struct active_shard* sh,
                  struct stream_metrics* metrics)
{
  if (!sink->wait_fence)
    return;
  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  sink->wait_fence(sink, sh->footer_io_done);
  if (metrics)
    accumulate_metric_ms(&metrics->footer_buffer_stall,
                         (float)(platform_toc(&clock) * 1000.0),
                         0,
                         0);
}

static int
ensure_writer(struct shard_sink* sink,
              const struct shard_drainer* drain,
              struct shard_state* ss,
              struct active_shard* sh,
              const struct shard_drain_command* command)
{
  if (sh->writer)
    return 0;
  sh->writer = sink->open(sink, command->level, command->flat_shard);
  if (!sh->writer)
    return 1;
  if (sh->writer->presize) {
    uint64_t capacity = ss->shard_file_capacity;
    if (capacity > 0 && drain->host->policy == HOST_DELIVERY_INDEXED_PADDED) {
      const uint64_t gap_count = ss->chunks_per_shard_append - 1;
      const uint64_t max_gap = drain->shard_alignment - 1;
      if (gap_count > 0 && max_gap > (UINT64_MAX - capacity) / gap_count)
        capacity = 0;
      else
        capacity += gap_count * max_gap;
    }
    if (sh->writer->presize(sh->writer, capacity))
      return 1;
  }
  return 0;
}

int
deliver_host_batch(struct host_batch* host,
                   struct shard_state* const* shards_by_lod,
                   struct shard_sink* sink,
                   size_t shard_alignment,
                   size_t* out_bytes,
                   struct stream_metrics* metrics)
{
  if (!host || !shards_by_lod || !sink ||
      host->shard_alignment != shard_alignment)
    return 1;

  struct shard_drainer drain;
  memset(&drain, 0, sizeof(drain));
  if (shard_drain_begin(&drain, host, shards_by_lod))
    return 1;

  size_t total_bytes = 0;
  struct shard_drain_command command;
  int next = 0;
  while ((next = shard_drain_next(&drain, &command)) > 0) {
    struct shard_state* ss = shards_by_lod[command.level];
    struct active_shard* sh = &ss->shards[command.inner_shard];
    int accepted = 0;

    if (command.kind == SHARD_DRAIN_DATA && command.physical_bytes == 0 &&
        drain.host->policy != HOST_DELIVERY_FIXED_TAIL) {
      // Empty indexed updates commit their logical epoch without creating a
      // physical file. Fixed updates may be holding a nonempty sub-page tail;
      // opening the writer lets a later stream flush finalize that tail.
      accepted = 1;
    } else {
      if (ensure_writer(sink, &drain, ss, sh, &command))
        goto Error;

      if (command.kind == SHARD_DRAIN_DATA && command.physical_bytes == 0) {
        accepted = 1;
      } else {

        if (command.kind == SHARD_DRAIN_FOOTER) {
          if (command.buffer_lease.kind == SHARD_DRAIN_LEASE_FOOTER)
            wait_footer_fence(sink, sh, metrics);
          if (shard_drain_prepare(&drain, &command))
            goto Error;
        }

        if (command.kind == SHARD_DRAIN_DATA ||
            command.kind == SHARD_DRAIN_FOOTER) {
          int wr = 1;
          if (command.direct_write_eligible && sh->writer->write_direct) {
            wr = sh->writer->write_direct(sh->writer,
                                          command.file_offset,
                                          command.source_begin,
                                          command.source_end);
            if (!wr && command.kind == SHARD_DRAIN_FOOTER && sink->record_fence)
              sh->footer_io_done = sink->record_fence(sink);
          } else {
            wr = sh->writer->write(sh->writer,
                                   command.file_offset,
                                   command.source_begin,
                                   command.source_end);
          }
          if (wr)
            goto Error;
          if (command.physical_bytes > SIZE_MAX - total_bytes)
            goto Error;
          total_bytes += command.physical_bytes;
          accepted = 1;
        } else if (command.kind == SHARD_DRAIN_TRUNCATE) {
          accepted =
            !sh->writer->truncate ||
            sh->writer->truncate(sh->writer, command.logical_size) == 0;
        } else if (command.kind == SHARD_DRAIN_FINALIZE) {
          accepted = sh->writer->finalize(sh->writer) == 0;
        }
      }
    }

    if (!accepted || shard_drain_accept(&drain, &command))
      goto Error;

    if (metrics && command.counts_physical_update) {
      metrics->shard_padding_logical_payload_bytes +=
        command.logical_payload_bytes;
      metrics->shard_padding_internal_bytes += command.internal_padding_bytes;
      metrics->shard_padding_physical_update_count++;
      if (command.counts_padded_update)
        metrics->shard_padding_padded_update_count++;
    }
    if (command.closes_generation && sink->record_fence) {
      ss->finalized_fence = sink->record_fence(sink);
      ss->fence_pending = 1;
    }
  }
  if (next < 0)
    goto Error;
  if (out_bytes)
    *out_bytes = total_bytes;
  shard_drain_destroy(&drain);
  return 0;

Error:
  shard_drain_abort(&drain);
  shard_drain_destroy(&drain);
  return 1;
}
