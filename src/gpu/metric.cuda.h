/// PRIVATE: never include in other headers.
#pragma once

#include "util/metric.h"
#include <cuda.h>

// Accumulate a completed interval. Returns 1 without accumulating while the
// interval is still outstanding, leaving the sample for the caller to retry.
static inline int
accumulate_metric_cu_if_ready(struct stream_metric* m,
                              CUevent start,
                              CUevent end,
                              size_t input_bytes,
                              size_t output_bytes)
{
  float ms = 0;
  if (cuEventQuery(end) != CUDA_SUCCESS)
    return 1;
  if (cuEventElapsedTime(&ms, start, end) != CUDA_SUCCESS)
    return 1;
  accumulate_metric_ms(m, ms, input_bytes, output_bytes);
  return 0;
}
