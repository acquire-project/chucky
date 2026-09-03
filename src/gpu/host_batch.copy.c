#include "gpu/host_batch.copy.h"

#include "gpu/prelude.cuda.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

#define D2H_TRY(err_flag, name, call)                                          \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      handle_curesult(LOG_ERROR, _r, __FILE__, __LINE__, (name));              \
      (err_flag) = 1;                                                          \
    }                                                                          \
  } while (0)

static int
copy_state_reserve_spans(struct d2h_copy_state* copy_state, size_t count)
{
  if (count <= copy_state->span_capacity)
    return 0;
  CHECK_MUL_OVERFLOW(Error, count, sizeof(*copy_state->spans), SIZE_MAX);
  struct d2h_transfer_span* p = (struct d2h_transfer_span*)realloc(
    copy_state->spans, count * sizeof(*copy_state->spans));
  CHECK(Error, p);
  copy_state->spans = p;
  copy_state->span_capacity = count;
  return 0;

Error:
  return 1;
}

static int
enqueue_metadata(struct d2h_copy_state* copy_state, CUstream stream)
{
  CHECK(Error,
        copy_state->batch.layout.total_batch_covering <=
          SIZE_MAX - copy_state->batch.layout.nlod);
  const size_t n = (size_t)copy_state->batch.layout.total_batch_covering +
                   copy_state->batch.layout.nlod;
  if (n == 0)
    return 0;
  CHECK_MUL_OVERFLOW(Error, n, sizeof(size_t), SIZE_MAX);
  const size_t metadata_bytes = n * sizeof(size_t);
  CHECK(Error, metadata_bytes <= SIZE_MAX / 2);

  int err = 0;
  D2H_TRY(err,
          "cuMemcpyDtoHAsync",
          cuMemcpyDtoHAsync(copy_state->slot->h_offsets,
                            (CUdeviceptr)copy_state->slot->d_offsets,
                            metadata_bytes,
                            stream));
  if (!err)
    D2H_TRY(err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(copy_state->slot->h_permuted_sizes,
                              (CUdeviceptr)copy_state->slot->d_permuted_sizes,
                              metadata_bytes,
                              stream));
  if (!err)
    copy_state->host.transfer.metadata_bytes_transferred = 2 * metadata_bytes;
  return err;

Error:
  return 1;
}

static int
enqueue_payload_spans(struct d2h_copy_state* copy_state,
                      CUstream stream,
                      CUevent start)
{
  int err = 0;
  if (cuEventRecord(start, stream) != CUDA_SUCCESS)
    err = 1;
  copy_state->payload_start = start;
  copy_state->host.transfer.payload_bytes_transferred = 0;
  copy_state->host.transfer.payload_copy_count = 0;

  for (size_t i = 0; i < copy_state->span_count && !err; ++i) {
    const struct d2h_transfer_span* span = &copy_state->spans[i];
    if (span->bytes == 0)
      continue;
    D2H_TRY(err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(
              (uint8_t*)copy_state->output.data + span->host_offset,
              (CUdeviceptr)copy_state->slot->d_aggregated + span->device_offset,
              span->bytes,
              stream));
    if (!err) {
      copy_state->host.transfer.payload_bytes_transferred += span->bytes;
      copy_state->host.transfer.payload_copy_count++;
    }
  }

  // One completion record covers every non-empty span.  It is also emitted on
  // failure so delivery and slot-reuse paths cannot strand a generation.
  if (gpu_pool_release_consume(copy_state->batch.aggregate_pool,
                               copy_state->batch.slot_index,
                               stream)) {
    err = 1;
  } else {
    copy_state->aggregate_released = 1;
  }
  return err;
}

static void
release_output(struct d2h_copy_state* copy_state)
{
  if (copy_state->host.output_group) {
    struct host_output_group* group = copy_state->host.output_group;
    copy_state->host.output_group = NULL;
    host_output_group_seal(group);
  } else if (copy_state->output.group) {
    host_output_group_seal(copy_state->output.group);
  }
  copy_state->output = (struct host_output){ 0 };
}

static int
release_output_after_payload(struct host_batch_copy* copy,
                             struct d2h_copy_state* copy_state,
                             int slot_index)
{
  if (copy_state->payload_started) {
    int complete = 0;
    if (copy_state->aggregate_released)
      complete = gpu_pool_host_acquire_consume(
                   copy_state->batch.host_pool, slot_index, NULL) == 0;
    if (!complete) {
      const CUresult result = cuStreamSynchronize(copy->payload_stream);
      if (result != CUDA_SUCCESS) {
        handle_curesult(
          LOG_ERROR, result, __FILE__, __LINE__, "cuStreamSynchronize");
        return 1;
      }
    }
    copy_state->payload_started = 0;
  }
  release_output(copy_state);
  return 0;
}

static int
fixed_begin(struct d2h_copy_state* copy_state)
{
  return aggregate_fixed_host_index(&copy_state->batch.layout,
                                    copy_state->batch.level_layouts,
                                    copy_state->batch.fixed_chunk_bytes,
                                    copy_state->slot->h_offsets,
                                    copy_state->slot->h_permuted_sizes);
}

static int
fixed_prepare_payload(struct host_batch_copy* copy,
                      struct d2h_copy_state* copy_state)
{
  struct gpu_pool_view lease;
  CHECK(Error,
        gpu_pool_acquire_consume(copy_state->batch.aggregate_pool,
                                 copy_state->batch.slot_index,
                                 copy->payload_stream,
                                 &lease) == 0);
  copy_state->slot = lease.p;
  copy_state->aggregate_acquired = 1;
  return 0;

Error:
  return 1;
}

static int
variable_begin(struct host_batch_copy* copy,
               struct d2h_copy_state* copy_state,
               CUstream stream)
{
  const int fc = copy_state->batch.slot_index;
  int err = 0;
  struct gpu_pool_view lease;
  CHECK(Error,
        gpu_pool_acquire_consume(
          copy_state->batch.aggregate_pool, fc, stream, &lease) == 0);
  copy_state->slot = lease.p;
  copy_state->aggregate_acquired = 1;
  D2H_TRY(
    err, "cuEventRecord", cuEventRecord(copy->metadata_copy_start[fc], stream));
  if (!err)
    err = enqueue_metadata(copy_state, stream);
  if (gpu_pool_release_produce(copy_state->batch.index_pool, fc, stream))
    err = 1;
  if (err) {
    if (gpu_pool_release_consume(
          copy_state->batch.aggregate_pool, fc, stream)) {
      err = 1;
    } else {
      copy_state->aggregate_released = 1;
    }
  }
  return err;

Error:
  return 1;
}

static int
variable_prepare_payload(struct host_batch_copy* copy,
                         struct d2h_copy_state* copy_state)
{
  const int fc = copy_state->batch.slot_index;
  struct gpu_pool_view index;
  if (gpu_pool_host_acquire_consume_split(copy_state->batch.index_pool,
                                          fc,
                                          GPU_EDGE_AGG_DONE,
                                          copy->aggregate_wait,
                                          copy->metadata_wait,
                                          &index))
    return 1;
  copy_state->slot = index.p;
  return 0;
}

int
host_batch_copy_init(struct host_batch_copy* copy,
                     enum aggregate_size_kind size_kind,
                     struct gpu_ordering* ordering,
                     CUstream payload_stream,
                     CUstream seed_stream)
{
  memset(copy, 0, sizeof(*copy));
  copy->size_kind = size_kind;
  copy->ordering = ordering;
  copy->payload_stream = payload_stream;
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail, cuEventCreate(&copy->metadata_copy_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&copy->payload_event[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(copy->metadata_copy_start[fc], seed_stream));
    CU(Fail, cuEventRecord(copy->payload_event[fc], seed_stream));
  }
  return 0;

Fail:
  host_batch_copy_destroy(copy);
  return 1;
}

void
host_batch_copy_set_wait_metrics(struct host_batch_copy* copy,
                                 struct stream_metric* aggregate_wait,
                                 struct stream_metric* metadata_wait)
{
  copy->aggregate_wait = aggregate_wait;
  copy->metadata_wait = metadata_wait;
}

void
host_batch_copy_destroy(struct host_batch_copy* copy)
{
  if (!copy)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    cu_event_destroy(copy->metadata_copy_start[fc]);
    cu_event_destroy(copy->payload_event[fc]);
    free(copy->state[fc].spans);
    release_output(&copy->state[fc]);
    host_batch_destroy(&copy->state[fc].host);
  }
  memset(copy, 0, sizeof(*copy));
}

int
host_batch_copy_begin(struct host_batch_copy* copy,
                      const struct aggregate_batch* batch,
                      CUstream metadata_stream)
{
  CHECK(Error, copy && batch);
  CHECK(Error, batch->slot_index == 0 || batch->slot_index == 1);
  CHECK(Error, batch->size_kind == copy->size_kind);
  struct d2h_copy_state* copy_state = &copy->state[batch->slot_index];
  CHECK(Error,
        copy_state->status == D2H_COPY_EMPTY ||
          copy_state->status == D2H_COPY_HOST_READY ||
          copy_state->status == D2H_COPY_ERROR);

  CHECK(Error,
        release_output_after_payload(copy, copy_state, batch->slot_index) == 0);
  copy_state->batch = *batch;
  copy_state->slot = gpu_pool_at(batch->aggregate_pool, batch->slot_index, 0).p;
  copy_state->span_count = 0;
  copy_state->host.run_count = 0;
  copy_state->host.transfer = (struct d2h_transfer_statistics){ 0 };
  copy_state->output = (struct host_output){ 0 };
  copy_state->aggregate_acquired = 0;
  copy_state->aggregate_released = 0;
  copy_state->payload_started = 0;
  copy_state->status = batch->size_kind == AGGREGATE_FIXED_SIZE
                         ? D2H_COPY_PAYLOAD_PENDING
                         : D2H_COPY_METADATA_PENDING;
  const int begin_error = batch->size_kind == AGGREGATE_FIXED_SIZE
                            ? fixed_begin(copy_state)
                            : variable_begin(copy, copy_state, metadata_stream);
  if (begin_error) {
    (void)host_batch_copy_cancel(copy, batch->slot_index);
    copy_state->status = D2H_COPY_ERROR;
    return 1;
  }
  return 0;

Error:
  return 1;
}

int
host_batch_copy_finish(struct host_batch_copy* copy,
                       int slot_index,
                       struct shard_state* const* shards_by_level,
                       size_t shard_alignment,
                       struct host_batch** out)
{
  CHECK(Error, copy && shards_by_level && out);
  CHECK(Error, slot_index == 0 || slot_index == 1);
  struct d2h_copy_state* copy_state = &copy->state[slot_index];
  CHECK(Error,
        copy_state->status == D2H_COPY_METADATA_PENDING ||
          copy_state->status == D2H_COPY_PAYLOAD_PENDING);
  const int prepare_error = copy->size_kind == AGGREGATE_FIXED_SIZE
                              ? fixed_prepare_payload(copy, copy_state)
                              : variable_prepare_payload(copy, copy_state);
  CHECK(Error, prepare_error == 0);
  CHECK(Error, copy_state->batch.output_pool);
  CHECK(Error,
        host_output_pool_acquire(copy_state->batch.output_pool,
                                 &copy_state->output) == 0);

  size_t capacity_bound = 0;
  size_t run_capacity = 0;
  const enum host_batch_storage storage = host_batch_storage_select(
    copy_state->batch.size_kind == AGGREGATE_FIXED_SIZE, shard_alignment);
  CHECK(Error,
        host_batch_capacity(copy_state->batch.level_layouts,
                            copy_state->batch.active_count_by_level,
                            copy_state->batch.layout.nlod,
                            storage,
                            shard_alignment,
                            &capacity_bound,
                            &run_capacity) == 0);
  CHECK(Error, capacity_bound <= copy_state->output.capacity);
  CHECK(Error, copy_state_reserve_spans(copy_state, run_capacity) == 0);
  CHECK(Error,
        host_batch_build(&copy_state->host,
                         copy_state->output.data,
                         copy_state->output.capacity,
                         copy_state->slot->h_offsets,
                         copy_state->slot->h_permuted_sizes,
                         &copy_state->batch.layout,
                         copy_state->batch.level_layouts,
                         shards_by_level,
                         copy_state->batch.active_count_by_level,
                         storage,
                         shard_alignment,
                         copy_state->spans,
                         copy_state->span_capacity,
                         &copy_state->span_count) == 0);
  copy_state->host.output_group = copy_state->output.group;
  copy_state->output.group = NULL;
  copy_state->payload_started = 1;
  copy_state->status = D2H_COPY_PAYLOAD_PENDING;
  const int payload_error = enqueue_payload_spans(
    copy_state, copy->payload_stream, copy->payload_event[slot_index]);
  CHECK(Error, payload_error == 0);
  CHECK(Error,
        gpu_pool_host_acquire_consume(
          copy_state->batch.host_pool, slot_index, NULL) == 0);
  copy_state->payload_started = 0;
  *out = &copy_state->host;
  copy_state->status = D2H_COPY_HOST_READY;
  return 0;

Error:
  if (copy && slot_index >= 0 && slot_index < 2) {
    (void)host_batch_copy_cancel(copy, slot_index);
    copy->state[slot_index].status = D2H_COPY_ERROR;
  }
  return 1;
}

int
host_batch_copy_cancel(struct host_batch_copy* copy, int slot_index)
{
  CHECK(Error, copy && (slot_index == 0 || slot_index == 1));
  struct d2h_copy_state* copy_state = &copy->state[slot_index];
  if (copy_state->status == D2H_COPY_EMPTY ||
      copy_state->status == D2H_COPY_HOST_READY) {
    CHECK(Error,
          release_output_after_payload(copy, copy_state, slot_index) == 0);
    copy_state->status = D2H_COPY_ERROR;
    return 0;
  }

  if (copy_state->aggregate_released) {
    CHECK(Error,
          release_output_after_payload(copy, copy_state, slot_index) == 0);
    copy_state->status = D2H_COPY_ERROR;
    return 0;
  }

  if (copy_state->batch.size_kind == AGGREGATE_VARIABLE_SIZE &&
      copy_state->aggregate_acquired) {
    // Retire metadata first: it reads arrays sharing this aggregate slot.
    (void)gpu_pool_host_acquire_consume(
      copy_state->batch.index_pool, slot_index, NULL);
  } else if (!copy_state->aggregate_acquired) {
    struct gpu_pool_view lease;
    CHECK(Error,
          gpu_pool_acquire_consume(copy_state->batch.aggregate_pool,
                                   slot_index,
                                   copy->payload_stream,
                                   &lease) == 0);
    copy_state->aggregate_acquired = 1;
  }

  CHECK(Error,
        gpu_pool_release_consume(copy_state->batch.aggregate_pool,
                                 slot_index,
                                 copy->payload_stream) == 0);
  copy_state->aggregate_released = 1;
  CHECK(Error, release_output_after_payload(copy, copy_state, slot_index) == 0);
  copy_state->status = D2H_COPY_ERROR;
  return 0;

Error:
  if (copy && slot_index >= 0 && slot_index < 2) {
    struct d2h_copy_state* failed = &copy->state[slot_index];
    (void)release_output_after_payload(copy, failed, slot_index);
    failed->status = D2H_COPY_ERROR;
  }
  return 1;
}
