#include "gpu/stream.ingest.h"

#include "gpu/prelude.cuda.h"
#include "gpu/transpose.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"

#include <string.h>

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
  if (!pool || n < (2u << 20)) {
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
    CU(Fail, cuEventCreate(&stage->slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(stage->slot[i].t_h2d_start, compute));
    CU(Fail, cuEventRecord(stage->slot[i].t_scatter_start, compute));
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
    cu_event_destroy(ss->t_scatter_start);
  }
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
  return 0;

Error:
  return 1;
}

int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct tile_stream_layout* layout,
                        const struct tile_stream_layout_gpu* layout_gpu,
                        struct gpu_pool_view pool_epoch,
                        uint64_t* cursor,
                        size_t bpe,
                        CUstream h2d,
                        CUstream compute)
{
  if (bpe == 0)
    return 0;

  const uint64_t elements = stage->bytes_written / bpe;
  if (elements == 0)
    return 0;

  const int idx = stage->current;
  struct staging_slot* ss = &stage->slot[idx];

  ss->dispatched_bytes = stage->bytes_written;

  struct gpu_pool_view d_in;
  CHECK(Error, dispatch_h2d(stage, idx, h2d, &d_in) == 0);

  // Scatter into chunk pool
  CHECK(Error,
        gpu_pool_acquire_consume(&stage->d_pool, idx, compute, &d_in) == 0);
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  transpose(gpu_pool_view_d(pool_epoch),
            gpu_pool_view_d(d_in),
            stage->bytes_written,
            (uint8_t)bpe,
            *cursor,
            layout->lifted_rank,
            layout_gpu->d_lifted_shape,
            layout_gpu->d_lifted_strides,
            compute);
  CHECK(Error, gpu_pool_release_consume(&stage->d_pool, idx, compute) == 0);

  *cursor += elements;
  stage->current ^= 1;
  return 0;

Error:
  return 1;
}

int
ingest_dispatch_multiscale(struct staging_state* stage,
                           CUdeviceptr d_linear,
                           uint64_t epoch_elements,
                           uint64_t* cursor,
                           size_t bpe,
                           CUstream h2d,
                           CUstream compute)
{
  if (bpe == 0)
    return 0;

  const uint64_t elements = stage->bytes_written / bpe;
  if (elements == 0)
    return 0;

  const int idx = stage->current;
  struct staging_slot* ss = &stage->slot[idx];

  ss->dispatched_bytes = stage->bytes_written;

  struct gpu_pool_view d_in;
  CHECK(Error, dispatch_h2d(stage, idx, h2d, &d_in) == 0);

  // Copy raw input to linear epoch buffer for LOD downsampling
  CHECK(Error,
        gpu_pool_acquire_consume(&stage->d_pool, idx, compute, &d_in) == 0);
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  {
    uint64_t epoch_offset = (*cursor % epoch_elements) * bpe;
    CU(Error,
       cuMemcpyDtoDAsync(d_linear + epoch_offset,
                         gpu_pool_view_d(d_in),
                         elements * bpe,
                         compute));
  }
  CHECK(Error, gpu_pool_release_consume(&stage->d_pool, idx, compute) == 0);

  *cursor += elements;
  stage->current ^= 1;
  return 0;

Error:
  return 1;
}
