#ifndef TEST_METRIC_CHECK_H
#define TEST_METRIC_CHECK_H

#include "log/log.h"
#include "types.stream.h"

// A stage whose measurements never arrive keeps count 0, and the bench report
// drops a zero-count row instead of printing zeros. So a stage that was never
// measured looks like a stage that was never run. Checking the count is what
// tells the two apart.
//
// Only the count and a duration above zero are checked. A threshold on the
// duration would be flaky on a shared machine, and the failure this guards
// against is a measurement that never arrives at all.
static inline int
metric_arrived(const struct stream_metric* m, int expected_count)
{
  const char* name = m->name ? m->name : "(unnamed)";
  if (m->count != expected_count) {
    log_error(
      "  %s: expected %d measurements, got %d", name, expected_count, m->count);
    return 0;
  }
  if (expected_count > 0 && !(m->ms > 0.0f)) {
    log_error(
      "  %s: %d measurements totalling %g ms", name, m->count, (double)m->ms);
    return 0;
  }
  return 1;
}

// The same check without pinning the number, for whole-stream runs where how
// many measurements a stage takes depends on how the schedule split the work.
static inline int
metric_arrived_at_least_once(const struct stream_metric* m)
{
  const char* name = m->name ? m->name : "(unnamed)";
  if (m->count <= 0) {
    log_error("  %s: no measurement arrived", name);
    return 0;
  }
  if (!(m->ms > 0.0f)) {
    log_error(
      "  %s: %d measurements totalling %g ms", name, m->count, (double)m->ms);
    return 0;
  }
  return 1;
}

#endif // TEST_METRIC_CHECK_H
