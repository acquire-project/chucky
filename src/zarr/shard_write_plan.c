#include "zarr/shard_write_plan.h"

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
write_context(struct shard_write_plan* plan,
              const struct host_batch_run** out_run,
              struct shard_state** out_ss,
              struct active_shard** out_sh)
{
  if (!plan || !plan->host || !plan->shards_by_level ||
      plan->run_index >= plan->host->run_count)
    return 1;
  const struct host_batch_run* run = &plan->host->runs[plan->run_index];
  if (run->level >= plan->host->nlod || !run->data)
    return 1;
  struct shard_state* ss = plan->shards_by_level[run->level];
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
finish_run(struct shard_write_plan* plan)
{
  plan->run_index++;
  plan->phase = SHARD_WRITE_PHASE_RUN;
  plan->data_physical_bytes = 0;
  plan->footer_remainder_bytes = 0;
  plan->footer_logical_bytes = 0;
  plan->footer_physical_bytes = 0;
}

static void
advance_nonfinal(struct shard_state* ss, const struct host_batch_run* run)
{
  if (run->ends_generation_run)
    ss->epoch_in_shard += run->active_count;
}

static int
start_run(struct shard_write_plan* plan)
{
  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (write_context(plan, &run, &ss, &sh))
    return 1;
  if (run->page_size != plan->shard_alignment)
    return 1;

  const enum host_batch_storage storage = plan->host->storage;
  const size_t alignment = plan->shard_alignment;
  if ((storage == HOST_BATCH_PAGE_PADDED && alignment == 0) ||
      (storage == HOST_BATCH_PACKED && alignment != 0))
    return 1;
  if (storage != HOST_BATCH_FIXED_SIZE && sh->tail_bytes != 0)
    return 1;
  if (storage == HOST_BATCH_FIXED_SIZE && alignment == 0 &&
      run->tail_bytes != 0)
    return 1;

  size_t total = run->payload_bytes;
  if (storage == HOST_BATCH_FIXED_SIZE && alignment > 0) {
    if (run->tail_bytes != sh->tail_bytes ||
        checked_add_size(run->tail_bytes, run->payload_bytes, &total))
      return 1;
  } else if (run->tail_bytes != 0) {
    return 1;
  }

  plan->run_start_cursor = sh->data_cursor;
  plan->data_physical_bytes = run->payload_bytes;
  plan->footer_remainder_bytes = 0;
  if (alignment > 0 && storage != HOST_BATCH_PACKED) {
    if (run->finalizes) {
      plan->data_physical_bytes = (total / alignment) * alignment;
      plan->footer_remainder_bytes = total - plan->data_physical_bytes;
    } else if (storage == HOST_BATCH_FIXED_SIZE) {
      plan->data_physical_bytes = (total / alignment) * alignment;
    } else if (checked_align_size(
                 run->payload_bytes, alignment, &plan->data_physical_bytes)) {
      return 1;
    }
  }

  if (sh->data_cursor > UINT64_MAX - plan->data_physical_bytes)
    return 1;
  uint64_t payload_file_offset = plan->run_start_cursor;
  if (storage == HOST_BATCH_FIXED_SIZE && alignment > 0) {
    if (payload_file_offset > UINT64_MAX - run->tail_bytes)
      return 1;
    payload_file_offset += run->tail_bytes;
  }
  if (validate_run_index(run, ss, payload_file_offset))
    return 1;
  if (!run->finalizes && storage == HOST_BATCH_FIXED_SIZE && alignment > 0) {
    const size_t remainder = total - plan->data_physical_bytes;
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
        checked_add_size(plan->footer_remainder_bytes, index_bytes, &logical) ||
        checked_add_size(logical, 4, &logical))
      return 1;
    plan->footer_logical_bytes = logical;
    plan->footer_physical_bytes = logical;
    if (alignment > 0 &&
        checked_align_size(logical, alignment, &plan->footer_physical_bytes))
      return 1;
    if (sh->data_cursor + plan->data_physical_bytes >
        UINT64_MAX - plan->footer_logical_bytes)
      return 1;
    plan->phase = plan->data_physical_bytes > 0 ? SHARD_WRITE_PHASE_DATA
                                                : SHARD_WRITE_PHASE_FOOTER;
    return 0;
  }

  // Empty variable-size updates still yield a zero-byte DATA command. The
  // executor accepts it without opening a writer, which keeps every index/epoch
  // transition behind shard_write_accept while emitting no padding or write.
  plan->phase = SHARD_WRITE_PHASE_DATA;
  return 0;
}

static int
command_common(struct shard_write_plan* plan,
               struct shard_write_command* command,
               enum shard_write_kind kind,
               const struct host_batch_run* run)
{
  if (plan->next_serial == UINT64_MAX)
    return 1;
  memset(command, 0, sizeof(*command));
  command->kind = kind;
  command->serial = ++plan->next_serial;
  command->level = run->level;
  command->inner_shard = run->inner_shard;
  command->flat_shard = run->flat_shard;
  return 0;
}

static int
command_matches(const struct shard_write_command* a,
                const struct shard_write_command* b)
{
  return a->kind == b->kind && a->serial == b->serial && a->level == b->level &&
         a->inner_shard == b->inner_shard && a->flat_shard == b->flat_shard &&
         a->file_offset == b->file_offset && a->source == b->source &&
         a->write_size == b->write_size &&
         a->truncate_size == b->truncate_size &&
         a->payload_bytes == b->payload_bytes &&
         a->padding_bytes == b->padding_bytes &&
         a->counts_shard_update == b->counts_shard_update &&
         a->counts_padded_update == b->counts_padded_update &&
         a->closes_generation == b->closes_generation;
}

int
shard_write_begin(struct shard_write_plan* plan,
                  struct host_batch* host,
                  struct shard_state* const* shards_by_level)
{
  if (!plan || !host || !shards_by_level ||
      (host->run_count > 0 && !host->runs) || host->nlod == 0 ||
      host->nlod > LOD_MAX_LEVELS)
    return 1;
  memset(plan, 0, sizeof(*plan));
  plan->host = host;
  plan->shards_by_level = shards_by_level;
  plan->shard_alignment = host->shard_alignment;
  plan->phase = SHARD_WRITE_PHASE_RUN;
  if (host->shard_alignment == 0 ? host->storage != HOST_BATCH_FIXED_SIZE &&
                                     host->storage != HOST_BATCH_PACKED
                                 : host->storage != HOST_BATCH_FIXED_SIZE &&
                                     host->storage != HOST_BATCH_PAGE_PADDED) {
    shard_write_abort(plan);
    return 1;
  }
  return 0;
}

int
shard_write_next(struct shard_write_plan* plan,
                 struct shard_write_command* command)
{
  if (!plan || !command || plan->failed || plan->pending)
    return -1;

  while (plan->phase == SHARD_WRITE_PHASE_RUN) {
    if (plan->run_index == plan->host->run_count)
      return 0;
    if (start_run(plan)) {
      shard_write_abort(plan);
      return -1;
    }
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (write_context(plan, &run, &ss, &sh)) {
    shard_write_abort(plan);
    return -1;
  }

  if (plan->phase == SHARD_WRITE_PHASE_DATA) {
    if (command_common(plan, command, SHARD_WRITE_DATA, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->source = run->data;
    command->write_size = plan->data_physical_bytes;
    if (run->payload_bytes > 0) {
      command->payload_bytes = run->payload_bytes;
      command->counts_shard_update = 1;
      if (!run->finalizes && plan->host->storage == HOST_BATCH_PAGE_PADDED) {
        command->padding_bytes = plan->data_physical_bytes - run->payload_bytes;
        command->counts_padded_update = command->padding_bytes > 0;
      }
    }
  } else if (plan->phase == SHARD_WRITE_PHASE_FOOTER) {
    if (command_common(plan, command, SHARD_WRITE_FOOTER, run))
      goto Error;
    command->file_offset = sh->data_cursor;
    command->write_size = plan->footer_physical_bytes;
    if (plan->shard_alignment > 0) {
      if (!sh->footer_buf ||
          plan->footer_physical_bytes > ss->footer_capacity) {
        shard_write_abort(plan);
        return -1;
      }
      command->source = sh->footer_buf;
    }
    if (plan->data_physical_bytes == 0 && run->payload_bytes > 0) {
      command->payload_bytes = run->payload_bytes;
      command->counts_shard_update = 1;
    }
  } else if (plan->phase == SHARD_WRITE_PHASE_TRUNCATE) {
    if (command_common(plan, command, SHARD_WRITE_TRUNCATE, run))
      goto Error;
    command->truncate_size = sh->data_cursor + plan->footer_logical_bytes;
  } else if (plan->phase == SHARD_WRITE_PHASE_FINALIZE) {
    if (command_common(plan, command, SHARD_WRITE_FINALIZE, run))
      goto Error;
    command->closes_generation = run->ends_generation_run;
  } else {
    shard_write_abort(plan);
    return -1;
  }

  plan->current = *command;
  plan->pending = 1;
  plan->prepared = command->kind != SHARD_WRITE_FOOTER;
  if (command->kind == SHARD_WRITE_FOOTER && plan->shard_alignment == 0 &&
      shard_write_prepare(plan, command))
    return -1;
  return 1;

Error:
  shard_write_abort(plan);
  return -1;
}

int
shard_write_prepare(struct shard_write_plan* plan,
                    struct shard_write_command* command)
{
  if (!plan || !command || plan->failed || !plan->pending ||
      command->serial != plan->current.serial ||
      command->kind != plan->current.kind)
    return 1;
  if (command->kind != SHARD_WRITE_FOOTER) {
    *command = plan->current;
    return 0;
  }
  if (plan->prepared) {
    *command = plan->current;
    return 0;
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  size_t index_bytes = 0;
  if (write_context(plan, &run, &ss, &sh) ||
      checked_index_bytes(ss, &index_bytes)) {
    shard_write_abort(plan);
    return 1;
  }

  uint8_t* dst = sh->footer_buf;
  if (plan->shard_alignment == 0) {
    plan->transient_footer = (uint8_t*)malloc(plan->footer_physical_bytes);
    if (!plan->transient_footer) {
      shard_write_abort(plan);
      return 1;
    }
    dst = plan->transient_footer;
  }
  if (!dst) {
    shard_write_abort(plan);
    return 1;
  }

  if (plan->footer_remainder_bytes > 0)
    memcpy(
      dst, run->data + plan->data_physical_bytes, plan->footer_remainder_bytes);
  uint8_t* index_dst = dst + plan->footer_remainder_bytes;
  memcpy(index_dst, sh->index, index_bytes);
  uint64_t payload_file_offset = plan->run_start_cursor;
  if (plan->host->storage == HOST_BATCH_FIXED_SIZE && plan->shard_alignment > 0)
    payload_file_offset += run->tail_bytes;
  patch_footer_index(index_dst, run, payload_file_offset);
  const uint32_t crc = crc32c(index_dst, index_bytes);
  memcpy(index_dst + index_bytes, &crc, sizeof(crc));
  if (plan->footer_physical_bytes > plan->footer_logical_bytes)
    memset(dst + plan->footer_logical_bytes,
           0,
           plan->footer_physical_bytes - plan->footer_logical_bytes);

  plan->current.source = dst;
  plan->prepared = 1;
  *command = plan->current;
  return 0;
}

int
shard_write_accept(struct shard_write_plan* plan,
                   const struct shard_write_command* command)
{
  if (!plan || !command || plan->failed || !plan->pending || !plan->prepared)
    return 1;
  if (!command_matches(command, &plan->current)) {
    shard_write_abort(plan);
    return 1;
  }

  const struct host_batch_run* run = NULL;
  struct shard_state* ss = NULL;
  struct active_shard* sh = NULL;
  if (write_context(plan, &run, &ss, &sh)) {
    shard_write_abort(plan);
    return 1;
  }
  if (command->kind == SHARD_WRITE_DATA) {
    sh->data_cursor += plan->data_physical_bytes;
    if (run->finalizes) {
      if (plan->shard_alignment == 0) {
        commit_run_index(sh, run, plan->run_start_cursor);
      }
      plan->phase = SHARD_WRITE_PHASE_FOOTER;
    } else {
      uint64_t payload_file_offset = plan->run_start_cursor;
      if (plan->host->storage == HOST_BATCH_FIXED_SIZE &&
          plan->shard_alignment > 0)
        payload_file_offset += run->tail_bytes;
      commit_run_index(sh, run, payload_file_offset);
      if (plan->host->storage == HOST_BATCH_FIXED_SIZE &&
          plan->shard_alignment > 0) {
        const size_t total = run->tail_bytes + run->payload_bytes;
        const size_t remainder = total - plan->data_physical_bytes;
        if (remainder > 0)
          memcpy(
            sh->tail_buf, run->data + plan->data_physical_bytes, remainder);
        sh->tail_bytes = remainder;
      } else {
        sh->tail_bytes = 0;
      }
      advance_nonfinal(ss, run);
      finish_run(plan);
    }
  } else if (command->kind == SHARD_WRITE_FOOTER) {
    free(plan->transient_footer);
    plan->transient_footer = NULL;
    if (plan->shard_alignment > 0) {
      uint64_t payload_file_offset = plan->run_start_cursor;
      if (plan->host->storage == HOST_BATCH_FIXED_SIZE)
        payload_file_offset += run->tail_bytes;
      commit_run_index(sh, run, payload_file_offset);
    }
    sh->tail_bytes = 0;
    plan->phase = SHARD_WRITE_PHASE_TRUNCATE;
  } else if (command->kind == SHARD_WRITE_TRUNCATE) {
    plan->phase = SHARD_WRITE_PHASE_FINALIZE;
  } else if (command->kind == SHARD_WRITE_FINALIZE) {
    size_t index_bytes = 0;
    if (checked_index_bytes(ss, &index_bytes)) {
      shard_write_abort(plan);
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
    finish_run(plan);
  } else {
    shard_write_abort(plan);
    return 1;
  }

  plan->pending = 0;
  plan->prepared = 0;
  memset(&plan->current, 0, sizeof(plan->current));
  return 0;
}

void
shard_write_abort(struct shard_write_plan* plan)
{
  if (!plan)
    return;
  free(plan->transient_footer);
  plan->transient_footer = NULL;
  plan->pending = 0;
  plan->prepared = 0;
  plan->failed = 1;
}

void
shard_write_destroy(struct shard_write_plan* plan)
{
  if (!plan)
    return;
  free(plan->transient_footer);
  memset(plan, 0, sizeof(*plan));
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
              const struct shard_write_plan* plan,
              struct shard_state* ss,
              struct active_shard* sh,
              const struct shard_write_command* command)
{
  if (sh->writer)
    return 0;
  sh->writer = sink->open(sink, command->level, command->flat_shard);
  if (!sh->writer)
    return 1;
  if (sh->writer->presize) {
    uint64_t capacity = ss->shard_file_capacity;
    if (capacity > 0 && plan->host->storage == HOST_BATCH_PAGE_PADDED) {
      const uint64_t gap_count = ss->chunks_per_shard_append - 1;
      const uint64_t max_gap = plan->shard_alignment - 1;
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
                   struct shard_state* const* shards_by_level,
                   struct shard_sink* sink,
                   size_t* out_bytes,
                   struct stream_metrics* metrics)
{
  if (!host || !shards_by_level || !sink)
    return 1;

  struct shard_write_plan plan;
  memset(&plan, 0, sizeof(plan));
  if (shard_write_begin(&plan, host, shards_by_level))
    return 1;

  size_t total_bytes = 0;
  struct shard_write_command command;
  int next = 0;
  while ((next = shard_write_next(&plan, &command)) > 0) {
    struct shard_state* ss = shards_by_level[command.level];
    struct active_shard* sh = &ss->shards[command.inner_shard];
    int accepted = 0;

    if (command.kind == SHARD_WRITE_DATA && command.write_size == 0 &&
        plan.host->storage != HOST_BATCH_FIXED_SIZE) {
      // Empty variable-size updates commit their logical epoch without creating
      // a physical file. Fixed updates may be holding a nonempty sub-page tail;
      // opening the writer lets a later stream flush finalize that tail.
      accepted = 1;
    } else {
      if (ensure_writer(sink, &plan, ss, sh, &command))
        goto Error;

      if (command.kind == SHARD_WRITE_DATA && command.write_size == 0) {
        accepted = 1;
      } else {

        if (command.kind == SHARD_WRITE_FOOTER) {
          if (plan.shard_alignment > 0)
            wait_footer_fence(sink, sh, metrics);
          if (shard_write_prepare(&plan, &command))
            goto Error;
        }

        if (command.kind == SHARD_WRITE_DATA ||
            command.kind == SHARD_WRITE_FOOTER) {
          const uint8_t* end = command.source + command.write_size;
          const int transient_footer =
            command.kind == SHARD_WRITE_FOOTER && plan.shard_alignment == 0;
          int use_direct =
            sh->writer->write_direct != NULL && !transient_footer;
          if (use_direct && plan.shard_alignment > 0) {
            const size_t alignment = plan.shard_alignment;
            use_direct = (uintptr_t)command.source % alignment == 0 &&
                         command.file_offset % alignment == 0 &&
                         command.write_size % alignment == 0;
          }
          int wr = 1;
          if (use_direct) {
            wr = sh->writer->write_direct(
              sh->writer, command.file_offset, command.source, end);
            if (!wr && command.kind == SHARD_WRITE_FOOTER && sink->record_fence)
              sh->footer_io_done = sink->record_fence(sink);
          } else {
            wr = sh->writer->write(
              sh->writer, command.file_offset, command.source, end);
          }
          if (wr)
            goto Error;
          if (command.write_size > SIZE_MAX - total_bytes)
            goto Error;
          total_bytes += command.write_size;
          accepted = 1;
        } else if (command.kind == SHARD_WRITE_TRUNCATE) {
          accepted =
            !sh->writer->truncate ||
            sh->writer->truncate(sh->writer, command.truncate_size) == 0;
        } else if (command.kind == SHARD_WRITE_FINALIZE) {
          accepted = sh->writer->finalize(sh->writer) == 0;
        }
      }
    }

    if (!accepted || shard_write_accept(&plan, &command))
      goto Error;

    if (metrics && command.counts_shard_update) {
      metrics->shard_padding_logical_payload_bytes += command.payload_bytes;
      metrics->shard_padding_internal_bytes += command.padding_bytes;
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
  shard_write_destroy(&plan);
  return 0;

Error:
  shard_write_abort(&plan);
  shard_write_destroy(&plan);
  return 1;
}
