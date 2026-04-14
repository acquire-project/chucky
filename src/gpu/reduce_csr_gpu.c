#include "gpu/reduce_csr_gpu.h"

#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"

#include <stdint.h>
#include <string.h>

int
reduce_csr_gpu_alloc(struct reduce_csr_gpu* csr,
                     uint64_t src_total,
                     uint64_t dst_total)
{
  memset(csr, 0, sizeof(*csr));
  csr->batch_count = 1;
  csr->dst_segment_size = dst_total;
  csr->src_lod_count = src_total;

  if (src_total == 0 || dst_total == 0)
    return 0;

  CU(Fail, cuMemAlloc(&csr->starts, (dst_total + 1) * sizeof(uint64_t)));
  CU(Fail, cuMemAlloc(&csr->indices, src_total * sizeof(uint64_t)));
  return 0;

Fail:
  reduce_csr_gpu_free(csr);
  return 1;
}

int
reduce_csr_gpu_build(struct reduce_csr_gpu* csr,
                     const struct level_dims* src,
                     const struct level_dims* dst,
                     CUstream stream)
{
  if (csr->src_lod_count == 0 || csr->dst_segment_size == 0)
    return 0;
  return lod_build_csr_gpu(csr->starts, csr->indices, src, dst, stream);
}

void
reduce_csr_gpu_free(struct reduce_csr_gpu* csr)
{
  if (!csr)
    return;
  if (csr->starts)
    cuMemFree(csr->starts);
  if (csr->indices)
    cuMemFree(csr->indices);
  memset(csr, 0, sizeof(*csr));
}
