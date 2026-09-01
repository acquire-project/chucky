#include "gpu/d2h.materializer.h"

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

struct d2h_materializer_ops
{
  int (*begin)(struct d2h_materializer*, struct d2h_ticket*, CUstream);
  int (*prepare_payload)(struct d2h_materializer*, struct d2h_ticket*);
};

static int
ticket_reserve_spans(struct d2h_ticket* ticket, size_t count)
{
  if (count <= ticket->span_capacity)
    return 0;
  CHECK_MUL_OVERFLOW(Error, count, sizeof(*ticket->spans), SIZE_MAX);
  struct d2h_transfer_span* p = (struct d2h_transfer_span*)realloc(
    ticket->spans, count * sizeof(*ticket->spans));
  CHECK(Error, p);
  ticket->spans = p;
  ticket->span_capacity = count;
  return 0;

Error:
  return 1;
}

static int
enqueue_metadata(struct d2h_ticket* ticket, CUstream stream)
{
  CHECK(Error,
        ticket->batch.layout.total_batch_covering <=
          SIZE_MAX - ticket->batch.nlod);
  const size_t n =
    (size_t)ticket->batch.layout.total_batch_covering + ticket->batch.nlod;
  if (n == 0)
    return 0;
  CHECK_MUL_OVERFLOW(Error, n, sizeof(size_t), SIZE_MAX);
  const size_t metadata_bytes = n * sizeof(size_t);
  CHECK(Error, metadata_bytes <= SIZE_MAX / 2);

  int err = 0;
  D2H_TRY(err,
          "cuMemcpyDtoHAsync",
          cuMemcpyDtoHAsync(ticket->slot->h_offsets,
                            (CUdeviceptr)ticket->slot->d_offsets,
                            metadata_bytes,
                            stream));
  if (!err)
    D2H_TRY(err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(ticket->slot->h_permuted_sizes,
                              (CUdeviceptr)ticket->slot->d_permuted_sizes,
                              metadata_bytes,
                              stream));
  if (!err)
    ticket->host.transfer.metadata_bytes_transferred = 2 * metadata_bytes;
  return err;

Error:
  return 1;
}

static int
enqueue_payload_spans(struct d2h_materializer* materializer,
                      struct d2h_ticket* ticket,
                      CUstream stream,
                      CUevent start,
                      int record_start)
{
  int err = 0;
  if (record_start && cuEventRecord(start, stream) != CUDA_SUCCESS)
    err = 1;
  ticket->payload_start = start;
  ticket->host.transfer.payload_bytes_transferred = 0;
  ticket->host.transfer.payload_copy_count = 0;

  for (size_t i = 0; i < ticket->span_count && !err; ++i) {
    const struct d2h_transfer_span* span = &ticket->spans[i];
    if (span->bytes == 0)
      continue;
    D2H_TRY(err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(
              (uint8_t*)ticket->slot->h_aggregated + span->host_offset,
              (CUdeviceptr)ticket->slot->d_aggregated + span->device_offset,
              span->bytes,
              stream));
    if (!err) {
      ticket->host.transfer.payload_bytes_transferred += span->bytes;
      ticket->host.transfer.payload_copy_count++;
    }
  }

  // One completion record covers every non-empty span.  It is also emitted on
  // failure so host/drain and slot-reuse paths cannot strand a generation.
  if (gpu_pool_release_consume(
        ticket->batch.aggregate_pool, ticket->batch.slot_index, stream)) {
    err = 1;
  } else {
    ticket->aggregate_released = 1;
  }
  (void)materializer;
  return err;
}

static int
fixed_begin(struct d2h_materializer* materializer,
            struct d2h_ticket* ticket,
            CUstream stream)
{
  (void)materializer;
  (void)stream;
  return aggregate_fixed_host_index(&ticket->batch.layout,
                                    ticket->batch.per_lod_layouts,
                                    ticket->batch.fixed_chunk_bytes,
                                    ticket->slot->h_offsets,
                                    ticket->slot->h_permuted_sizes);
}

static int
fixed_prepare_payload(struct d2h_materializer* materializer,
                      struct d2h_ticket* ticket)
{
  struct gpu_pool_view lease;
  CHECK(Error,
        gpu_pool_acquire_consume(ticket->batch.aggregate_pool,
                                 ticket->batch.slot_index,
                                 materializer->payload_stream,
                                 &lease) == 0);
  ticket->slot = lease.p;
  ticket->aggregate_acquired = 1;
  return 0;

Error:
  return 1;
}

static int
indexed_begin(struct d2h_materializer* materializer,
              struct d2h_ticket* ticket,
              CUstream stream)
{
  const int fc = ticket->batch.slot_index;
  int err = 0;
  struct gpu_pool_view lease;
  CHECK(Error,
        gpu_pool_acquire_consume(
          ticket->batch.aggregate_pool, fc, stream, &lease) == 0);
  ticket->slot = lease.p;
  ticket->aggregate_acquired = 1;
  D2H_TRY(err,
          "cuEventRecord",
          cuEventRecord(materializer->metadata_copy_start[fc], stream));
  if (!err)
    err = enqueue_metadata(ticket, stream);
  if (gpu_pool_release_produce(ticket->batch.index_pool, fc, stream))
    err = 1;
  if (err) {
    if (gpu_pool_release_consume(ticket->batch.aggregate_pool, fc, stream)) {
      err = 1;
    } else {
      ticket->aggregate_released = 1;
    }
  }
  return err;

Error:
  return 1;
}

static int
indexed_prepare_payload(struct d2h_materializer* materializer,
                        struct d2h_ticket* ticket)
{
  const int fc = ticket->batch.slot_index;
  struct gpu_pool_view index;
  if (gpu_pool_host_acquire_consume_split(ticket->batch.index_pool,
                                          fc,
                                          GPU_EDGE_AGG_DONE,
                                          materializer->aggregate_ready_stall,
                                          materializer->metadata_ready_stall,
                                          &index))
    return 1;
  ticket->slot = index.p;
  return 0;
}

static const struct d2h_materializer_ops FIXED_OPS = {
  .begin = fixed_begin,
  .prepare_payload = fixed_prepare_payload,
};

static const struct d2h_materializer_ops INDEXED_OPS = {
  .begin = indexed_begin,
  .prepare_payload = indexed_prepare_payload,
};

int
d2h_materializer_init(struct d2h_materializer* materializer,
                      enum device_aggregate_extent_kind extent_kind,
                      struct gpu_ordering* ord,
                      CUstream payload_stream,
                      CUstream seed_stream)
{
  memset(materializer, 0, sizeof(*materializer));
  materializer->extent_kind = extent_kind;
  materializer->ops =
    extent_kind == DEVICE_AGGREGATE_FIXED_EXTENT ? &FIXED_OPS : &INDEXED_OPS;
  materializer->ord = ord;
  materializer->payload_stream = payload_stream;
  for (int fc = 0; fc < 2; ++fc) {
    CU(Fail,
       cuEventCreate(&materializer->metadata_copy_start[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&materializer->payload_event[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(materializer->metadata_copy_start[fc], seed_stream));
    CU(Fail, cuEventRecord(materializer->payload_event[fc], seed_stream));
  }
  return 0;

Fail:
  d2h_materializer_destroy(materializer);
  return 1;
}

void
d2h_materializer_attach_metadata_stalls(struct d2h_materializer* materializer,
                                        struct stream_metric* aggregate_ready,
                                        struct stream_metric* metadata_ready)
{
  materializer->aggregate_ready_stall = aggregate_ready;
  materializer->metadata_ready_stall = metadata_ready;
}

void
d2h_materializer_destroy(struct d2h_materializer* materializer)
{
  if (!materializer)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    cu_event_destroy(materializer->metadata_copy_start[fc]);
    cu_event_destroy(materializer->payload_event[fc]);
    free(materializer->ticket[fc].spans);
    host_batch_destroy(&materializer->ticket[fc].host);
  }
  memset(materializer, 0, sizeof(*materializer));
}

int
d2h_materialize_begin(struct d2h_materializer* materializer,
                      const struct device_aggregate_batch* batch,
                      CUstream metadata_stream)
{
  CHECK(Error, materializer && batch);
  CHECK(Error, batch->slot_index == 0 || batch->slot_index == 1);
  CHECK(Error, batch->extent_kind == materializer->extent_kind);
  struct d2h_ticket* ticket = &materializer->ticket[batch->slot_index];
  CHECK(Error,
        ticket->state == D2H_TICKET_EMPTY ||
          ticket->state == D2H_TICKET_HOST_READY ||
          ticket->state == D2H_TICKET_ERROR);

  ticket->batch = *batch;
  ticket->slot = gpu_pool_at(batch->aggregate_pool, batch->slot_index, 0).p;
  ticket->span_count = 0;
  ticket->host.run_count = 0;
  ticket->host.slot_lifetime = ticket->slot;
  ticket->host.transfer = (struct d2h_transfer_statistics){ 0 };
  ticket->aggregate_acquired = 0;
  ticket->aggregate_released = 0;
  ticket->state = batch->extent_kind == DEVICE_AGGREGATE_FIXED_EXTENT
                    ? D2H_TICKET_PAYLOAD_PENDING
                    : D2H_TICKET_METADATA_PENDING;
  if (materializer->ops->begin(materializer, ticket, metadata_stream)) {
    (void)d2h_materialize_cancel(materializer, batch->slot_index);
    ticket->state = D2H_TICKET_ERROR;
    return 1;
  }
  return 0;

Error:
  return 1;
}

int
d2h_materialize_finish(struct d2h_materializer* materializer,
                       int slot_index,
                       const struct d2h_host_placement* placement,
                       struct host_batch** out)
{
  CHECK(Error, materializer && placement && out);
  CHECK(Error, slot_index == 0 || slot_index == 1);
  struct d2h_ticket* ticket = &materializer->ticket[slot_index];
  CHECK(Error,
        ticket->state == D2H_TICKET_METADATA_PENDING ||
          ticket->state == D2H_TICKET_PAYLOAD_PENDING);
  CHECK(Error, materializer->ops->prepare_payload(materializer, ticket) == 0);

  size_t capacity_bound = 0;
  size_t run_capacity = 0;
  const enum host_delivery_policy policy = host_delivery_policy_select(
    ticket->batch.extent_kind == DEVICE_AGGREGATE_FIXED_EXTENT,
    placement->shard_alignment);
  CHECK(Error,
        host_batch_compact_capacity(placement->per_lod_layouts,
                                    ticket->batch.per_lod_n_active,
                                    ticket->batch.nlod,
                                    policy,
                                    placement->shard_alignment,
                                    &capacity_bound,
                                    &run_capacity) == 0);
  CHECK(Error, capacity_bound <= ticket->slot->host_capacity);
  CHECK(Error, ticket_reserve_spans(ticket, run_capacity) == 0);
  CHECK(Error,
        host_batch_build_compact(&ticket->host,
                                 ticket->slot->h_aggregated,
                                 ticket->slot->host_capacity,
                                 ticket->slot->h_offsets,
                                 ticket->slot->h_permuted_sizes,
                                 &ticket->batch.layout,
                                 placement->per_lod_layouts,
                                 placement->shards_by_lod,
                                 ticket->batch.per_lod_n_active,
                                 ticket->batch.nlod,
                                 policy,
                                 placement->shard_alignment,
                                 ticket->spans,
                                 ticket->span_capacity,
                                 &ticket->span_count,
                                 placement->slot_lifetime) == 0);
  ticket->state = D2H_TICKET_PAYLOAD_PENDING;
  const int payload_error =
    enqueue_payload_spans(materializer,
                          ticket,
                          materializer->payload_stream,
                          materializer->payload_event[slot_index],
                          1);
  CHECK(Error, payload_error == 0);
  CHECK(Error,
        gpu_pool_host_acquire_consume(
          ticket->batch.host_pool, slot_index, NULL) == 0);
  *out = &ticket->host;
  ticket->state = D2H_TICKET_HOST_READY;
  return 0;

Error:
  if (materializer && slot_index >= 0 && slot_index < 2) {
    (void)d2h_materialize_cancel(materializer, slot_index);
    materializer->ticket[slot_index].state = D2H_TICKET_ERROR;
  }
  return 1;
}

int
d2h_materialize_cancel(struct d2h_materializer* materializer, int slot_index)
{
  CHECK(Error, materializer && (slot_index == 0 || slot_index == 1));
  struct d2h_ticket* ticket = &materializer->ticket[slot_index];
  if (ticket->state == D2H_TICKET_EMPTY ||
      ticket->state == D2H_TICKET_HOST_READY) {
    ticket->state = D2H_TICKET_ERROR;
    return 0;
  }

  if (ticket->aggregate_released) {
    CHECK(Error,
          gpu_pool_host_acquire_consume(
            ticket->batch.host_pool, slot_index, NULL) == 0);
    ticket->state = D2H_TICKET_ERROR;
    return 0;
  }

  if (ticket->batch.extent_kind == DEVICE_AGGREGATE_INDEXED_EXTENT &&
      ticket->aggregate_acquired) {
    // Retire metadata first: it reads arrays sharing this aggregate slot.
    (void)gpu_pool_host_acquire_consume(
      ticket->batch.index_pool, slot_index, NULL);
  } else if (!ticket->aggregate_acquired) {
    struct gpu_pool_view lease;
    CHECK(Error,
          gpu_pool_acquire_consume(ticket->batch.aggregate_pool,
                                   slot_index,
                                   materializer->payload_stream,
                                   &lease) == 0);
    ticket->aggregate_acquired = 1;
  }

  CHECK(Error,
        gpu_pool_release_consume(ticket->batch.aggregate_pool,
                                 slot_index,
                                 materializer->payload_stream) == 0);
  ticket->aggregate_released = 1;
  (void)gpu_pool_host_acquire_consume(
    ticket->batch.host_pool, slot_index, NULL);
  ticket->state = D2H_TICKET_ERROR;
  return 0;

Error:
  if (materializer && slot_index >= 0 && slot_index < 2)
    materializer->ticket[slot_index].state = D2H_TICKET_ERROR;
  return 1;
}
