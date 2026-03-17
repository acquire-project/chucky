/// PRIVATE: never include in other headers.
#pragma once

#include "metric.h"
#include <cuda.h>

static inline void
accumulate_metric_cu(struct stream_metric* m,
                     CUevent start,
                     CUevent end,
                     size_t bytes)
{
  float ms = 0;
  cuEventElapsedTime(&ms, start, end);
  if (ms < 1e-2f)
    return; // skip bogus measurements from seeded events
  accumulate_metric_ms(m, ms, bytes);
}
