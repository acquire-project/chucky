#include "stream_internal.h"

#include "prelude.cuda.h"
#include "transpose.h"

int
ingest_dispatch_scatter(struct staging_state* stage,
                        const struct stream_layout* layout,
                        void* pool_epoch,
                        CUevent pool_ready,
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
  CU(Error, cuStreamWaitEvent(h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, h2d));
  CU(Error, cuMemcpyHtoDAsync(ss->d_in, ss->h_in, stage->bytes_written, h2d));
  CU(Error, cuEventRecord(ss->t_h2d_end, h2d));

  // Scatter into tile pool
  CU(Error, cuStreamWaitEvent(compute, ss->t_h2d_end, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  transpose((CUdeviceptr)pool_epoch,
            ss->d_in,
            stage->bytes_written,
            (uint8_t)bpe,
            *cursor,
            layout->lifted_rank,
            layout->d_lifted_shape,
            layout->d_lifted_strides,
            compute);
  CU(Error, cuEventRecord(ss->t_scatter_end, compute));

  CU(Error, cuEventRecord(pool_ready, compute));

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
  CU(Error, cuStreamWaitEvent(h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, h2d));
  CU(Error, cuMemcpyHtoDAsync(ss->d_in, ss->h_in, stage->bytes_written, h2d));
  CU(Error, cuEventRecord(ss->t_h2d_end, h2d));

  // Copy raw input to linear epoch buffer for LOD downsampling
  CU(Error, cuStreamWaitEvent(compute, ss->t_h2d_end, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, compute));
  {
    uint64_t epoch_offset = (*cursor % epoch_elements) * bpe;
    CU(Error,
       cuMemcpyDtoDAsync(
         d_linear + epoch_offset, ss->d_in, elements * bpe, compute));
  }
  CU(Error, cuEventRecord(ss->t_scatter_end, compute));

  *cursor += elements;
  stage->current ^= 1;
  return 0;

Error:
  return 1;
}
