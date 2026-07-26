/// PRIVATE: never include in other headers.
#pragma once

#include "util/metric.h"
#include <cuda.h>

// Accumulate only once the interval has finished on the device. Returns 1 while
// it is still outstanding so the caller can keep the sample and retry, rather
// than reading a zero and discarding it. No magnitude filter: the caller owns
// deciding which intervals are real, so a genuinely sub-10us stage still
// counts.
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
