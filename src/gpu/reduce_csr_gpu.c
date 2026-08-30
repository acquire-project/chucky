#include "gpu/reduce_csr_gpu.h"

#include "gpu/lod.h"
#include "gpu/prelude.cuda.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"

#include <stdint.h>
#include <string.h>

int
reduce_csr_gpu_alloc(struct reduce_csr_gpu* csr,
                     uint64_t src_total,
                     uint64_t dst_total)
{
  CHECK(Fail, csr);
  memset(csr, 0, sizeof(*csr));
  csr->batch_count = 1;
  csr->dst_segment_size = dst_total;
  csr->src_lod_count = src_total;

  CHECK(Fail, src_total > 0);
  CHECK(Fail, dst_total > 0);

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
  CHECK(Fail, csr);
  CHECK(Fail, src);
  CHECK(Fail, dst);

  CHECK_MUL_OVERFLOW(Fail, src->fixed_dims_count, src->lod_nelem, UINT64_MAX);
  const uint64_t src_total = src->fixed_dims_count * src->lod_nelem;
  CHECK(Fail, src_total > 0);

  CHECK_MUL_OVERFLOW(Fail, dst->fixed_dims_count, dst->lod_nelem, UINT64_MAX);
  const uint64_t dst_total = dst->fixed_dims_count * dst->lod_nelem;
  CHECK(Fail, dst_total > 0);

  CHECK(Fail, csr->src_lod_count == src_total);
  CHECK(Fail, csr->dst_segment_size == dst_total);
  CHECK(Fail, csr->src_lod_count > 0);
  CHECK(Fail, csr->dst_segment_size > 0);
  CHECK(Fail, csr->starts);
  CHECK(Fail, csr->indices);
  CHECK(Fail,
        lod_build_csr_gpu(csr->starts, csr->indices, src, dst, stream) == 0);
  return 0;

Fail:
  return 1;
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
