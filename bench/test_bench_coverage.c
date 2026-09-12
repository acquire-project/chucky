#include "bench_measurement.h"
#include "bench_report.h"

#include <float.h>
#include <stdio.h>

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int
main(void)
{
  struct bench_measurement run = {
    .boundary_bytes = { 1024, 2048, 1024 },
    .requested_warmup_s = 0.01,
    .target_duration_s = 0.01,
  };
  REQUIRE(!bench_warmup_ready(&run, 2048, 0.249));
  REQUIRE(!bench_warmup_ready(&run, 2047, 10));
  REQUIRE(bench_warmup_ready(&run, 2048, 0.25));
  run.requested_warmup_s = 2;
  REQUIRE(!bench_warmup_ready(&run, 4096, 1));
  REQUIRE(bench_warmup_ready(&run, 4096, 2));
  run.requested_warmup_s = 0;
  run.warmup_s = 0.25;
  run.warmup_bytes = 2048;
  run.append_s = 0.25;
  bench_measurement_count(&run, run.warmup_bytes);
  REQUIRE(run.generation_transitions == 0);
  bench_measurement_count(&run, run.warmup_bytes + 4095);
  REQUIRE(run.complete_batches == 3 && !bench_measurement_ready(&run));
  bench_measurement_count(&run, run.warmup_bytes + 4096);
  REQUIRE(run.complete_batches == 4 && run.batch_reuses == 2);
  REQUIRE(run.generation_transitions == 1 && !bench_measurement_ready(&run));
  bench_measurement_count(&run, run.warmup_bytes + 4097);
  REQUIRE(run.generation_transitions == 2 && bench_measurement_ready(&run));
  run.append_s = 0.249;
  REQUIRE(!bench_measurement_ready(&run));
  run.target_duration_s = 1;
  run.append_s = 0.999;
  REQUIRE(!bench_measurement_ready(&run));
  run.append_s = 1;
  REQUIRE(bench_measurement_ready(&run));
  run.elapsed_s = 1;
  run.drain_s = 0;
  REQUIRE(bench_measurement_covered(&run));
  run.elapsed_s = 1.1;
  run.drain_s = 0.1;
  REQUIRE(bench_measurement_covered(&run));
  run.elapsed_s = 1.2;
  run.drain_s = 0.2;
  REQUIRE(!bench_measurement_covered(&run));
  REQUIRE(bench_measurement_retry_duration(&run) == 3.6);
  run.append_s = 4;
  REQUIRE(bench_measurement_retry_duration(&run) == 8);
  run.append_s = DBL_MAX;
  REQUIRE(bench_measurement_retry_duration(&run) == 0);
  return 0;
}
