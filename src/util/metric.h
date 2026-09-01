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
    i = (int)(decades * APPEND_LATENCY_PER_DECADE);
    if (i >= APPEND_LATENCY_BUCKETS)
      i = APPEND_LATENCY_BUCKETS - 1;
  }
  m->append_ms_buckets[i]++;
  m->append_count++;
}

// The upper edge of bucket i, or the longest append seen when that is smaller.
// The topmost bucket has no upper edge, so it reports the longest append.
static inline float
append_bucket_ms(const struct stream_metrics* m, int i)
{
  if (i >= APPEND_LATENCY_BUCKETS - 1)
    return m->max_append_ms;
  const float edge =
    (float)(APPEND_LATENCY_MIN_MS *
            pow(10.0, (double)(i + 1) / APPEND_LATENCY_PER_DECADE));
  return edge < m->max_append_ms ? edge : m->max_append_ms;
}

// Duration within which the given fraction of appends completed. Never exceeds
// the longest append seen. Returns 0 when nothing was recorded.
static inline float
append_ms_at(const struct stream_metrics* m, double fraction)
{
  if (m->append_count == 0)
    return 0.0f;
  uint64_t want = (uint64_t)ceil(fraction * (double)m->append_count);
  if (want == 0)
    want = 1;
  uint64_t seen = 0;
  for (int i = 0; i < APPEND_LATENCY_BUCKETS; ++i) {
    seen += m->append_ms_buckets[i];
    if (seen >= want)
      return append_bucket_ms(m, i);
  }
  return m->max_append_ms;
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
  if (ms > m->max_ms)
    m->max_ms = ms;
  if (ms < m->best_ms) {
    m->best_ms = ms;
    m->best_input_bytes = (double)input_bytes;
    m->best_output_bytes = (double)output_bytes;
  }
}

static inline void
record_duration_ms(struct duration_stats* stats, float ms)
{
  stats->total_ms += ms;
  if (stats->count == 0 || ms < stats->min_ms)
    stats->min_ms = ms;
  if (stats->count == 0 || ms > stats->max_ms)
    stats->max_ms = ms;
  stats->count++;
}
