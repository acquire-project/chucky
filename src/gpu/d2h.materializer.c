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
  int (*finish)(struct d2h_materializer*, struct d2h_ticket*);
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

static size_t
metadata_count(const struct device_aggregate_batch* batch)
{
  return (size_t)batch->layout.total_batch_covering + batch->nlod;
}

static int
enqueue_metadata(struct d2h_ticket* ticket, CUstream stream)
{
  const size_t n = metadata_count(&ticket->batch);
  if (n == 0)
    return 0;

  int err = 0;
  D2H_TRY(err,
          "cuMemcpyDtoHAsync",
          cuMemcpyDtoHAsync(ticket->slot->h_offsets,
                            (CUdeviceptr)ticket->slot->d_offsets,
                            n * sizeof(size_t),
                            stream));
  if (!err)
    D2H_TRY(err,
            "cuMemcpyDtoHAsync",
            cuMemcpyDtoHAsync(ticket->slot->h_permuted_sizes,
                              (CUdeviceptr)ticket->slot->d_permuted_sizes,
                              n * sizeof(size_t),
                              stream));
  if (!err)
    ticket->host.transfer.metadata_bytes_transferred = 2 * n * sizeof(size_t);
  return err;
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
        ticket->batch.aggregate_pool, ticket->batch.slot_index, stream))
    err = 1;
  (void)materializer;
  return err;
}

static int
fixed_begin(struct d2h_materializer* materializer,
            struct d2h_ticket* ticket,
            CUstream stream)
{
  const int fc = ticket->batch.slot_index;
  int err =
    cuEventRecord(materializer->begin_event[fc], stream) != CUDA_SUCCESS;
  ticket->payload_start = materializer->begin_event[fc];
  if (enqueue_metadata(ticket, stream))
    err = 1;
  if (ticket_reserve_spans(ticket, 1))
    err = 1;
  if (!err && d2h_plan_legacy_spans(&ticket->batch.layout,
                                    ticket->batch.nlod,
                                    ticket->batch.per_lod_n_active,
                                    NULL,
                                    NULL,
                                    1,
                                    ticket->spans,
                                    ticket->span_capacity,
                                    &ticket->span_count))
    err = 1;
  if (err)
    ticket->span_count = 0;
  if (enqueue_payload_spans(
        materializer, ticket, stream, materializer->begin_event[fc], 0))
    err = 1;
  return err;
}

static int
fixed_finish(struct d2h_materializer* materializer, struct d2h_ticket* ticket)
{
  (void)materializer;
  struct gpu_pool_view host;
  return gpu_pool_host_acquire_consume(
    ticket->batch.host_pool, ticket->batch.slot_index, &host);
}

static int
indexed_begin(struct d2h_materializer* materializer,
              struct d2h_ticket* ticket,
              CUstream stream)
{
  (void)materializer;
  int err = enqueue_metadata(ticket, stream);
  if (gpu_pool_release_produce(
        ticket->batch.index_pool, ticket->batch.slot_index, stream))
    err = 1;
  if (err && gpu_pool_release_consume(
               ticket->batch.aggregate_pool, ticket->batch.slot_index, stream))
    err = 1;
  return err;
}

static int
indexed_plan_spans(struct d2h_ticket* ticket)
{
  CHECK(Error, ticket_reserve_spans(ticket, ticket->batch.nlod) == 0);
  return d2h_plan_legacy_spans(&ticket->batch.layout,
                               ticket->batch.nlod,
                               ticket->batch.per_lod_n_active,
                               ticket->slot->h_offsets,
                               ticket->slot->h_permuted_sizes,
                               0,
                               ticket->spans,
                               ticket->span_capacity,
                               &ticket->span_count);

Error:
  ticket->span_count = 0;
  return 1;
}

static int
indexed_finish(struct d2h_materializer* materializer, struct d2h_ticket* ticket)
{
  const int fc = ticket->batch.slot_index;
  struct gpu_pool_view index;
  if (gpu_pool_host_acquire_consume(ticket->batch.index_pool, fc, &index))
    return 1;
  ticket->slot = index.p;
  if (indexed_plan_spans(ticket))
    goto ReleaseError;
  ticket->state = D2H_TICKET_PAYLOAD_PENDING;
  if (enqueue_payload_spans(materializer,
                            ticket,
                            materializer->payload_stream,
                            materializer->payload_event[fc],
                            1))
    return 1;
  return gpu_pool_host_acquire_consume(ticket->batch.host_pool, fc, NULL);

ReleaseError:
  (void)gpu_pool_release_consume(
    ticket->batch.aggregate_pool, fc, materializer->payload_stream);
  return 1;
}

static const struct d2h_materializer_ops FIXED_OPS = {
  .begin = fixed_begin,
  .finish = fixed_finish,
};

static const struct d2h_materializer_ops INDEXED_OPS = {
  .begin = indexed_begin,
  .finish = indexed_finish,
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
    CU(Fail, cuEventCreate(&materializer->begin_event[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(materializer->begin_event[fc], seed_stream));
    CU(Fail, cuEventCreate(&materializer->payload_event[fc], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(materializer->payload_event[fc], seed_stream));
  }
  return 0;

Fail:
  d2h_materializer_destroy(materializer);
  return 1;
}

void
d2h_materializer_destroy(struct d2h_materializer* materializer)
{
  if (!materializer)
    return;
  for (int fc = 0; fc < 2; ++fc) {
    cu_event_destroy(materializer->begin_event[fc]);
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

  struct gpu_pool_view lease;
  CHECK(Error,
        gpu_pool_acquire_consume(
          batch->aggregate_pool, batch->slot_index, metadata_stream, &lease) ==
          0);

  ticket->batch = *batch;
  ticket->slot = lease.p;
  ticket->span_count = 0;
  ticket->host.run_count = 0;
  ticket->host.slot_lifetime = lease.p;
  ticket->host.transfer = (struct d2h_transfer_statistics){ 0 };
  ticket->state = batch->extent_kind == DEVICE_AGGREGATE_FIXED_EXTENT
                    ? D2H_TICKET_PAYLOAD_PENDING
                    : D2H_TICKET_METADATA_PENDING;
  if (materializer->ops->begin(materializer, ticket, metadata_stream)) {
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
  CHECK(Error, materializer->ops->finish(materializer, ticket) == 0);

  CHECK(Error,
        host_batch_build_legacy(&ticket->host,
                                ticket->slot->h_aggregated,
                                ticket->slot->h_offsets,
                                ticket->slot->h_permuted_sizes,
                                &ticket->batch.layout,
                                placement->per_lod_layouts,
                                placement->shards_by_lod,
                                ticket->batch.per_lod_n_active,
                                ticket->batch.nlod,
                                placement->slot_lifetime) == 0);
  *out = &ticket->host;
  ticket->state = D2H_TICKET_HOST_READY;
  return 0;

Error:
  if (materializer && slot_index >= 0 && slot_index < 2)
    materializer->ticket[slot_index].state = D2H_TICKET_ERROR;
  return 1;
}
