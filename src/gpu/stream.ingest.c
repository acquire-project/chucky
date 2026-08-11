#include "gpu/stream.ingest.h"

#include "gpu/metric.cuda.h"
#include "gpu/prelude.cuda.h"
#include "gpu/transpose.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"

#include <string.h>

// Splitting pays above ~24 KiB, where the ~1.4 us dispatch stops dominating,
// so 64 KiB keeps margin. Parked helpers cost 15 us, but a producer idle that
// long has slack to absorb it.
#define COPY_POOL_MIN_BYTES (64u << 10)

struct copy_slices
{
  uint8_t* dst;
  const uint8_t* src;
};

static void
copy_slice(size_t beg, size_t end, int tid, void* vctx)
{
  (void)tid;
  struct copy_slices* c = (struct copy_slices*)vctx;
  memcpy(c->dst + beg, c->src + beg, end - beg);
}

void
ingest_copy(struct threadpool* pool, void* dst, const void* src, size_t n)
{
  if (!pool || n < COPY_POOL_MIN_BYTES) {
    memcpy(dst, src, n);
    return;
  }
  struct copy_slices c = { (uint8_t*)dst, (const uint8_t*)src };
  threadpool_for_n(pool, n, copy_slice, &c);
}

int
ingest_init(struct staging_state* stage,
            size_t buffer_capacity_bytes,
            struct gpu_ordering* ord,
            CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  // Ordering events (h2d-done, scatter-done) are owned and seeded by ord;
  // only the timing-interval starts are created here.
  gpu_pool_init(&stage->d_pool,
                ord,
                GPU_EDGE_STAGING_H2D_DONE,
                GPU_EDGE_STAGING_SCATTER_DONE);
  gpu_pool_init(&stage->h_pool, ord, GPU_EDGE_COUNT, GPU_EDGE_STAGING_FREE);
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemHostAlloc(&stage->slot[i].h_in, buffer_capacity_bytes, 0));
    CU(Fail, cuMemAlloc(&stage->slot[i].d_in, buffer_capacity_bytes));
    gpu_pool_bind(&stage->h_pool, i, stage->slot[i].h_in);
    gpu_pool_bind(&stage->d_pool, i, (void*)(uintptr_t)stage->slot[i].d_in);
    CU(Fail, cuEventCreate(&stage->slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->slot[i].t_h2d_start, compute));
  }
  for (int i = 0; i < SCATTER_TIMING_SLOTS; ++i) {
    CU(Fail, cuEventCreate(&stage->timing[i].t_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&stage->timing[i].t_end, CU_EVENT_DEFAULT));
  }
  return 0;
Fail:
  ingest_destroy(stage);
  return 1;
}

void
ingest_destroy(struct staging_state* stage)
{
  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stage->slot[i];
    cu_mem_freehost(ss->h_in);
    cu_mem_free(ss->d_in);
    cu_event_destroy(ss->t_h2d_start);
  }
  for (int i = 0; i < SCATTER_TIMING_SLOTS; ++i) {
    cu_event_destroy(stage->timing[i].t_start);
    cu_event_destroy(stage->timing[i].t_end);
  }
}

void
ingest_collect_h2d_timing(struct staging_state* stage, struct stream_metric* m)
{
  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stage->slot[i];
    if (!ss->h2d_pending)
      continue;
    CUevent end =
      gpu_ordering_event(stage->d_pool.ord, GPU_EDGE_STAGING_H2D_DONE, i);
    if (accumulate_metric_cu_if_ready(m,
                                      ss->t_h2d_start,
                                      end,
                                      ss->dispatched_bytes,
                                      ss->dispatched_bytes) == 0)
      ss->h2d_pending = 0;
  }
}

void
ingest_collect_scatter_timing(struct staging_state* stage,
                              struct stream_metric* m)
{
  for (int i = 0; i < SCATTER_TIMING_SLOTS; ++i) {
    struct scatter_timing* st = &stage->timing[i];
    if (!st->pending)
      continue;
    if (accumulate_metric_cu_if_ready(
          m, st->t_start, st->t_end, st->bytes, st->bytes) == 0)
      st->pending = 0;
  }
}

// Claims the next measurement in the ring and opens it. An entry still
// outstanding has outlived its slack, so count it rather than lose it
// silently. Returns NULL when the start could not be recorded.
static struct scatter_timing*
timing_begin(struct staging_state* stage, uint64_t bytes, CUstream compute)
{
  struct scatter_timing* st = &stage->timing[stage->next_timing];
  if (st->pending)
    stage->scatter_samples_lost++;
  stage->next_timing = (stage->next_timing + 1) % SCATTER_TIMING_SLOTS;
  st->bytes = bytes;
  st->pending = 0;
  if (handle_curesult(LOG_ERROR,
                      cuEventRecord(st->t_start, compute),
                      __FILE__,
                      __LINE__,
                      "cuEventRecord"))
    return NULL;
  return st;
}

// The measurement only counts once both ends are recorded: an interval with one
// end reports whatever its events still hold.
static int
timing_end(struct scatter_timing* st, CUstream compute)
{
  CU(Fail, cuEventRecord(st->t_end, compute));
  st->pending = 1;
  return 0;

Fail:
  return 1;
}

static int
scatter_with_timing(struct staging_state* stage,
                    const struct tile_stream_layout* layout,
                    struct scatter_destination dst,
                    struct gpu_pool_view d_in,
                    uint64_t bytes,
                    uint64_t in_epoch,
                    size_t bpe,
                    CUstream compute)
{
  struct scatter_timing* st = timing_begin(stage, bytes, compute);
  CHECK_SILENT(Fail, st);
  CHECK_SILENT(Fail,
               transpose(gpu_pool_view_d(dst.first_epoch),
                         gpu_pool_view_d(d_in),
                         bytes,
                         (uint8_t)bpe,
                         in_epoch,
                         layout->epoch_elements,
                         dst.epoch_bytes,
                         layout->lifted_rank,
                         layout->lifted_shape,
                         layout->lifted_strides,
                         compute) == 0);
  return timing_end(st, compute);

Fail:
  return 1;
}

static int
copy_to_linear_with_timing(struct staging_state* stage,
                           CUdeviceptr d_linear,
                           struct gpu_pool_view d_in,
                           uint64_t bytes,
                           uint64_t epoch_offset_bytes,
                           CUstream compute)
{
  struct scatter_timing* st = timing_begin(stage, bytes, compute);
  CHECK_SILENT(Fail, st);
  CU(Fail,
     cuMemcpyDtoDAsync(
       d_linear + epoch_offset_bytes, gpu_pool_view_d(d_in), bytes, compute));
  return timing_end(st, compute);

Fail:
  return 1;
}

// The produce-acquire keeps the H2D copy from overwriting d_in while the
// prior generation's scatter still reads it; the release also frees h_in
// for host refill (STAGING_FREE aliases it).
static int
dispatch_h2d(struct staging_state* stage,
             int idx,
             CUstream h2d,
             struct gpu_pool_view* d_in)
{
  struct gpu_pool_view h_in;
  // h_in consume: ready is host call order (filled before this dispatch).
  CHECK(Error, gpu_pool_acquire_consume(&stage->h_pool, idx, h2d, &h_in) == 0);
  CHECK(Error, gpu_pool_acquire_produce(&stage->d_pool, idx, h2d, d_in) == 0);
  CU(Error, cuEventRecord(stage->slot[idx].t_h2d_start, h2d));
  CU(Error,
     cuMemcpyHtoDAsync(
       gpu_pool_view_d(*d_in), h_in.p, stage->bytes_written, h2d));
  CHECK(Error, gpu_pool_release_produce(&stage->d_pool, idx, h2d) == 0);
  stage->slot[idx].h2d_pending = 1;
  return 0;

Error:
  return 1;
}

int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct tile_stream_layout* layout,
                        struct scatter_destination dst,
                        uint64_t first_element,
                        size_t bpe,
                        CUstream h2d,
                        CUstream compute)
{
  if (bpe == 0)
    return 0;

  const uint64_t elements = stage->bytes_written / bpe;
  if (elements == 0)
    return 0;

  // A chunk position repeats every epoch, since the append dimensions do not
  // reach it.
  const uint64_t epoch_elements = layout->epoch_elements;
  const uint64_t in_epoch = first_element % epoch_elements;
  // Regions past the ones the caller acquired belong to the next pool slot.
  CHECK(Error, ceildiv(in_epoch + elements, epoch_elements) <= dst.epochs);

  const int idx = stage->current;
  struct staging_slot* ss = &stage->slot[idx];

  ss->dispatched_bytes = stage->bytes_written;

  struct gpu_pool_view d_in;
  CHECK(Error, dispatch_h2d(stage, idx, h2d, &d_in) == 0);

  // Scatter into chunk pool
  CHECK(Error,
        gpu_pool_acquire_consume(&stage->d_pool, idx, compute, &d_in) == 0);
  const int failed = scatter_with_timing(
    stage, layout, dst, d_in, ss->dispatched_bytes, in_epoch, bpe, compute);
  // Nothing reads the input either way, so hand the slot back before failing.
  CHECK(Error, gpu_pool_release_consume(&stage->d_pool, idx, compute) == 0);
  CHECK_SILENT(Error, !failed);

  stage->current ^= 1;
  return 0;

Error:
  return 1;
}

int
ingest_dispatch_multiscale(struct staging_state* stage,
                           CUdeviceptr d_linear,
                           uint64_t epoch_elements,
                           uint64_t first_element,
                           size_t bpe,
                           CUstream h2d,
                           CUstream compute)
{
  if (bpe == 0)
    return 0;

  const uint64_t elements = stage->bytes_written / bpe;
  if (elements == 0)
    return 0;

  const uint64_t in_epoch = first_element % epoch_elements;
  // d_linear holds one epoch; the copy below would run past it.
  CHECK(Error, in_epoch + elements <= epoch_elements);

  const int idx = stage->current;
  struct staging_slot* ss = &stage->slot[idx];

  ss->dispatched_bytes = stage->bytes_written;

  struct gpu_pool_view d_in;
  CHECK(Error, dispatch_h2d(stage, idx, h2d, &d_in) == 0);

  // Copy raw input to linear epoch buffer for LOD downsampling
  CHECK(Error,
        gpu_pool_acquire_consume(&stage->d_pool, idx, compute, &d_in) == 0);
  const int failed = copy_to_linear_with_timing(
    stage, d_linear, d_in, elements * bpe, in_epoch * bpe, compute);
  // Nothing reads the input either way, so hand the slot back before failing.
  CHECK(Error, gpu_pool_release_consume(&stage->d_pool, idx, compute) == 0);
  CHECK_SILENT(Error, !failed);

  stage->current ^= 1;
  return 0;

Error:
  return 1;
}
