#include "gpu/stream.ingest.h"

#include "gpu/prelude.cuda.h"
#include "gpu/transpose.h"
#include "util/prelude.h"

#include <string.h>

int
ingest_init(struct staging_state* stage,
            size_t buffer_capacity_bytes,
            struct gpu_ordering* ord,
            CUstream compute)
{
  memset(stage, 0, sizeof(*stage));
  stage->ord = ord;
  // Ordering events (h2d-done, scatter-done) are owned and seeded by ord;
  // only the timing-interval starts are created here.
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemHostAlloc(&stage->slot[i].h_in, buffer_capacity_bytes, 0));
    CU(Fail, cuMemAlloc(&stage->slot[i].d_in, buffer_capacity_bytes));
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

int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct tile_stream_layout* layout,
                        const struct tile_stream_layout_gpu* layout_gpu,
                        void* pool_epoch,
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

  // H2D — wait for prior scatter to finish reading d_in before overwriting
  CHECK(Error,
        gpu_edge_wait(
          stage->ord, GPU_EDGE_STAGING_SCATTER_DONE, idx, h2d) == 0);
  CU(Error, cuEventRecord(ss->t_h2d_start, h2d));
  CU(Error, cuMemcpyHtoDAsync(ss->d_in, ss->h_in, stage->bytes_written, h2d));
  CHECK(Error,
        gpu_edge_record(stage->ord, GPU_EDGE_STAGING_H2D_DONE, idx, h2d) == 0);

  // Scatter into chunk pool
  CHECK(Error,
        gpu_edge_wait(
          stage->ord, GPU_EDGE_STAGING_H2D_DONE, idx, compute) == 0);
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  transpose((CUdeviceptr)pool_epoch,
            ss->d_in,
            stage->bytes_written,
            (uint8_t)bpe,
            *cursor,
            layout->lifted_rank,
            layout_gpu->d_lifted_shape,
            layout_gpu->d_lifted_strides,
            compute);
  CHECK(Error,
        gpu_edge_record(
          stage->ord, GPU_EDGE_STAGING_SCATTER_DONE, idx, compute) == 0);

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

  // H2D — wait for prior d_linear copy to finish reading d_in
  CHECK(Error,
        gpu_edge_wait(
          stage->ord, GPU_EDGE_STAGING_SCATTER_DONE, idx, h2d) == 0);
  CU(Error, cuEventRecord(ss->t_h2d_start, h2d));
  CU(Error, cuMemcpyHtoDAsync(ss->d_in, ss->h_in, stage->bytes_written, h2d));
  CHECK(Error,
        gpu_edge_record(stage->ord, GPU_EDGE_STAGING_H2D_DONE, idx, h2d) == 0);

  // Copy raw input to linear epoch buffer for LOD downsampling
  CHECK(Error,
        gpu_edge_wait(
          stage->ord, GPU_EDGE_STAGING_H2D_DONE, idx, compute) == 0);
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  {
    uint64_t epoch_offset = (*cursor % epoch_elements) * bpe;
    CU(Error,
       cuMemcpyDtoDAsync(
         d_linear + epoch_offset, ss->d_in, elements * bpe, compute));
  }
  CHECK(Error,
        gpu_edge_record(
          stage->ord, GPU_EDGE_STAGING_SCATTER_DONE, idx, compute) == 0);

  *cursor += elements;
  stage->current ^= 1;
  return 0;

Error:
  return 1;
}
