#pragma once

#include <stdint.h>

struct bench_measurement;

extern const char bench_measurement_policy[];

int
bench_warmup_ready(const struct bench_measurement* run,
                   uint64_t accepted,
                   double elapsed_s);

void
bench_measurement_count(struct bench_measurement* run, uint64_t accepted);

int
bench_measurement_ready(const struct bench_measurement* run);

int
bench_measurement_covered(const struct bench_measurement* run);

// Plan another attempt with margin below the 10% drain ceiling. The next
// attempt must still pass the final check; this estimate is not a guarantee.
double
bench_measurement_retry_duration(const struct bench_measurement* run);
