#include "bench_measurement.h"
#include "bench_report.h"

#include <math.h>

const char bench_measurement_policy[] =
  "coverage-qualified-through-final-close-v2";

int
bench_warmup_ready(const struct bench_measurement* run,
                   uint64_t accepted,
                   double elapsed_s)
{
  return elapsed_s >= fmax(0.25, run->requested_warmup_s) &&
         run->boundary_bytes[0] && accepted / run->boundary_bytes[0] >= 2;
}

void
bench_measurement_count(struct bench_measurement* run, uint64_t accepted)
{
  run->input_bytes = accepted - run->warmup_bytes;
  run->complete_batches = run->input_bytes / run->boundary_bytes[0];
  // Two batch buffers on GPU; this lower bound also applies to CPU.
  run->batch_reuses = run->complete_batches > 2 ? run->complete_batches - 2 : 0;
  run->generation_transitions = run->input_bytes
                                  ? (accepted - 1) / run->boundary_bytes[1] -
                                      run->warmup_bytes / run->boundary_bytes[1]
                                  : 0;
}

int
bench_measurement_ready(const struct bench_measurement* run)
{
  return run->append_s >= fmax(0.25, run->target_duration_s) &&
         run->complete_batches >= 4 && run->generation_transitions >= 2;
}

int
bench_measurement_covered(const struct bench_measurement* run)
{
  return bench_warmup_ready(run, run->warmup_bytes, run->warmup_s) &&
         bench_measurement_ready(run) && run->drain_s >= 0 &&
         run->drain_s <= 0.1 * run->elapsed_s;
}

double
bench_measurement_retry_duration(const struct bench_measurement* run)
{
  // At the observed drain cost, 9x drain is the 10% boundary. Double that
  // allowance, and at least double the previous append window.
  const double next = fmax(2 * run->append_s, 18 * run->drain_s);
  return isfinite(next) ? next : 0;
}
