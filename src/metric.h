/// PRIVATE: never include in other headers.
#pragma once

#include <stddef.h>

struct stream_metric
{
  const char* name;
  float ms;            // cumulative
  float best_ms;       // best single measurement (1e30f = not yet measured)
  double input_bytes;  // cumulative bytes read by stage
  double output_bytes; // cumulative bytes written by stage
  int count;
};

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
