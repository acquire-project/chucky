#include "gpu/schedule.h"

#include "gpu/ordering.h"
#include "gpu/prelude.cuda.h"

int
gpu_streams_init(struct gpu_streams* s)
{
  CU(Fail, cuStreamCreate(&s->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->d2h, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&s->drain, CU_STREAM_NON_BLOCKING));
  return 0;
Fail:
  return 1;
}

void
gpu_streams_destroy(struct gpu_streams* s)
{
  cu_stream_destroy(s->h2d);
  cu_stream_destroy(s->compute);
  cu_stream_destroy(s->compress);
  cu_stream_destroy(s->d2h);
  cu_stream_destroy(s->drain);
}

void
gpu_streams_register(const struct gpu_streams* s, struct gpu_ordering* ord)
{
  gpu_ordering_register_stream(ord, GPU_STREAM_H2D, s->h2d);
  gpu_ordering_register_stream(ord, GPU_STREAM_COMPUTE, s->compute);
  gpu_ordering_register_stream(ord, GPU_STREAM_COMPRESS, s->compress);
  gpu_ordering_register_stream(ord, GPU_STREAM_D2H, s->d2h);
  gpu_ordering_register_stream(ord, GPU_STREAM_DRAIN, s->drain);
}
