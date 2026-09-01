#ifndef TEST_METRIC_CHECK_H
#define TEST_METRIC_CHECK_H

#include "log/log.h"
#include "types.stream.h"

// A stage that was never measured keeps count 0, and the report drops a
// zero-count row, so it reads as a stage that never ran. Count is the contract.
//
// Hold an interval that brackets work to a positive duration. The smallest here
// is 6 to 7 us against a timer resolving to about half of one. An interval that
// brackets a wait is the exception: a wait that never waited is honestly zero.

static inline int
metric_arrived(const struct stream_metric* m, int expected_count)
{
  if (m->count != expected_count) {
    log_error("  %s: expected %d measurements, got %d",
              m->name,
              expected_count,
              m->count);
    return 0;
  }
  // A total hides one backwards interval among good ones. The fastest single
  // measurement cannot: a negative is smaller than anything real.
  if (m->count > 0 && m->best_ms < 0.0f) {
    log_error(
      "  %s: a measurement ran backwards, %g ms", m->name, (double)m->best_ms);
    return 0;
  }
  if (m->count > 0 && m->max_ms < m->best_ms) {
    log_error("  %s: slowest measurement %g ms is below fastest %g ms",
              m->name,
              (double)m->max_ms,
              (double)m->best_ms);
    return 0;
  }
  return 1;
}

// Every one of the expected measurements took time the timer could resolve.
static inline int
metric_arrived_timed(const struct stream_metric* m, int expected_count)
{
  if (!metric_arrived(m, expected_count))
    return 0;
  if (!(expected_count > 0 && m->best_ms > 0.0f)) {
    log_error("  %s: a measurement took %g ms", m->name, (double)m->best_ms);
    return 0;
  }
  return 1;
}

// For whole-stream runs, where the schedule decides how many times a stage
// ran. Holding each of an unknown number to the timer would buy a flake.
static inline int
metric_any_arrived_timed(const struct stream_metric* m)
{
  if (m->count <= 0) {
    log_error("  %s: no measurement arrived", m->name);
    return 0;
  }
  if (!(m->ms > 0.0f)) {
    log_error("  %s: %d measurements totalling %g ms",
              m->name,
              m->count,
              (double)m->ms);
    return 0;
  }
  return 1;
}

static inline int
duration_any_arrived_timed(const char* name, const struct duration_stats* stats)
{
  if (stats->count == 0) {
    log_error("  %s: no measurement arrived", name);
    return 0;
  }
  if (!(stats->total_ms > 0.0f)) {
    log_error("  %s: %llu measurements totalling %g ms",
              name,
              (unsigned long long)stats->count,
              (double)stats->total_ms);
    return 0;
  }
  if (stats->min_ms < 0.0f || stats->max_ms < stats->min_ms) {
    log_error("  %s: invalid range %g to %g ms",
              name,
              (double)stats->min_ms,
              (double)stats->max_ms);
    return 0;
  }
  return 1;
}

// A ring that overwrote a measurement before anyone read it under-reports its
// stage, and this counter is the only trace.
static inline int
no_samples_lost(const char* ring, uint64_t lost)
{
  if (lost == 0)
    return 1;
  log_error("  %s ring wrapped %llu times", ring, (unsigned long long)lost);
  return 0;
}

#endif // TEST_METRIC_CHECK_H
