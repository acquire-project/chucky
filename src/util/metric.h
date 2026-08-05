/// PRIVATE: never include in other headers.
#pragma once

#include "types.stream.h"

#include <math.h>

#include <stddef.h>

struct stream_metric
mk_stream_metric(const char* name, enum metric_owner owner);

// Record one append's duration.
static inline void
record_append_ms(struct stream_metrics* m, float ms)
{
  int i = 0;
  if (ms > (float)APPEND_LATENCY_MIN_MS) {
    const double decades = log10((double)ms / APPEND_LATENCY_MIN_MS);
    i = (int)(decades * APPEND_LATENCY_PER_DECADE + 0.5);
    if (i >= APPEND_LATENCY_BUCKETS)
      i = APPEND_LATENCY_BUCKETS - 1;
  }
  m->append_ms_buckets[i]++;
  m->append_count++;
}

// Duration at or below which the given fraction of appends completed. Returns
// 0 when nothing was recorded.
static inline float
append_ms_at(const struct stream_metrics* m, double fraction)
{
  if (m->append_count == 0)
    return 0.0f;
  const uint64_t want = (uint64_t)(fraction * (double)m->append_count + 0.5);
  uint64_t seen = 0;
  for (int i = 0; i < APPEND_LATENCY_BUCKETS; ++i) {
    seen += m->append_ms_buckets[i];
    if (seen >= want || i == APPEND_LATENCY_BUCKETS - 1)
      return (float)(APPEND_LATENCY_MIN_MS *
                     pow(10.0, (double)i / APPEND_LATENCY_PER_DECADE));
  }
  return 0.0f;
}

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
  if (ms < m->best_ms) {
    m->best_ms = ms;
    m->best_input_bytes = (double)input_bytes;
    m->best_output_bytes = (double)output_bytes;
  }
}
