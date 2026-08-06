#include "chucky_log.h"
#include "util/metric.h"
#include "util/prelude.h"

#include <math.h>

static int
test_no_samples_reports_nothing(void)
{
  log_info("=== test_no_samples_reports_nothing ===");
  struct stream_metrics m = { 0 };
  CHECK(Fail, m.append_count == 0);
  CHECK(Fail, append_ms_at(&m, 0.5) == 0.0f);
  CHECK(Fail, append_ms_at(&m, 0.999) == 0.0f);
  return 0;
Fail:
  return 1;
}

static void
record(struct stream_metrics* m, float ms)
{
  if (ms > m->max_append_ms)
    m->max_append_ms = ms;
  record_append_ms(m, ms);
}

// Every sample must land in a bucket whose range contains it, or a reported
// time can come out above the longest append seen.
static int
test_every_sample_lands_in_a_containing_bucket(void)
{
  log_info("=== test_every_sample_lands_in_a_containing_bucket ===");
  for (int k = 0; k < 4000; ++k) {
    const float ms = (float)(APPEND_LATENCY_MIN_MS * pow(10.0, k / 400.0));
    struct stream_metrics m = { 0 };
    record(&m, ms);
    CHECK(Fail, m.append_count == 1);
    const float got = append_ms_at(&m, 1.0);
    CHECK(Fail, got >= ms * 0.999f);
    CHECK(Fail, got <= m.max_append_ms);
  }
  return 0;
Fail:
  return 1;
}

static int
test_no_percentile_exceeds_the_longest_append(void)
{
  log_info("=== test_no_percentile_exceeds_the_longest_append ===");
  const float samples[] = { 0.0004f, 0.02f,  0.13f,  1.0f,   2.51f,
                            17.8f,   60.74f, 68.54f, 144.31f };
  struct stream_metrics m = { 0 };
  for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i)
    record(&m, samples[i]);
  const double fractions[] = { 0.01, 0.5, 0.9, 0.99, 0.999, 1.0 };
  for (size_t i = 0; i < sizeof(fractions) / sizeof(fractions[0]); ++i) {
    const float got = append_ms_at(&m, fractions[i]);
    CHECK(Fail, got > 0.0f);
    CHECK(Fail, got <= m.max_append_ms);
  }
  return 0;
Fail:
  return 1;
}

// Asking for a fraction must not return a faster time than that fraction of
// appends actually achieved.
static int
test_fraction_is_not_rounded_down(void)
{
  log_info("=== test_fraction_is_not_rounded_down ===");
  for (int n = 1; n <= 40; ++n) {
    struct stream_metrics m = { 0 };
    for (int i = 0; i < n; ++i)
      record(&m, (float)(0.01 * (i + 1)));
    const float got = append_ms_at(&m, 0.9);
    uint64_t at_or_below = 0;
    for (int i = 0; i < APPEND_LATENCY_BUCKETS; ++i) {
      if (append_bucket_ms(&m, i) <= got)
        at_or_below += m.append_ms_buckets[i];
    }
    CHECK(Fail, at_or_below * 100 >= (uint64_t)n * 90);
  }
  return 0;
Fail:
  return 1;
}

static int
test_slowest_bucket_reports_the_longest_append(void)
{
  log_info("=== test_slowest_bucket_reports_the_longest_append ===");
  struct stream_metrics m = { 0 };
  record(&m, 1.0f);
  record(&m, 1.0e7f); // past the last bucket's lower edge
  CHECK(Fail, append_ms_at(&m, 1.0) == m.max_append_ms);
  return 0;
Fail:
  return 1;
}

static int
test_samples_below_the_floor_are_counted(void)
{
  log_info("=== test_samples_below_the_floor_are_counted ===");
  struct stream_metrics m = { 0 };
  record(&m, 0.0f);
  record(&m, -1.0f);
  record(&m, 1e-9f);
  CHECK(Fail, m.append_count == 3);
  CHECK(Fail, m.append_ms_buckets[0] == 3);
  return 0;
Fail:
  return 1;
}

int
main(void)
{
  int err = 0;
  err |= test_no_samples_reports_nothing();
  err |= test_every_sample_lands_in_a_containing_bucket();
  err |= test_no_percentile_exceeds_the_longest_append();
  err |= test_fraction_is_not_rounded_down();
  err |= test_slowest_bucket_reports_the_longest_append();
  err |= test_samples_below_the_floor_are_counted();
  if (err)
    log_error("FAILED");
  else
    log_info("PASSED");
  return err;
}
