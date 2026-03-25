/// PRIVATE: never include in other headers.
#pragma once

#include "types.stream.h"

#include <stddef.h>

struct stream_metric
mk_stream_metric(const char* name);

static inline void
accumulate_metric_ms(struct stream_metric* m,
                     float ms,
                     size_t input_bytes,
                     size_t output_bytes)
{
  m->ms += ms;
  m->input_bytes += input_bytes;
  m->output_bytes += output_bytes;
  m->count++;
  if (ms < m->best_ms)
    m->best_ms = ms;
}
