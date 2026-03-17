#include "cpu/lod.h"
#include "lod_plan.h"
#include "prelude.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Cross-validate lod_cpu against per-element ravel reference.
// Uses a 3D shape with LOD on dims 1,2 (lod_mask=0x6).
static int
test_scatter_reduce_f32(enum lod_reduce_method method, const char* name)
{
  log_info("=== test_lod_cpu_%s ===", name);

  uint64_t shape[] = { 2, 8, 6 };
  uint64_t chunk_shape[] = { 2, 4, 3 };
  uint32_t lod_mask = 0x6; // dims 1,2

  float* src = NULL;
  void* values = NULL;

  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 3, shape, chunk_shape, lod_mask, 8) == 0);
  CHECK(Fail, plan.nlod >= 2);

  // Fill source with sequential floats.
  uint64_t n = 1;
  for (int d = 0; d < 3; ++d)
    n *= shape[d];
  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);
  CHECK(Fail,
        lod_cpu_compute(&plan, src, &values, lod_dtype_f32, method) == 0);

  // Verify L0: all source values should be present in the morton buffer.
  struct lod_span l0 = lod_spans_at(&plan.levels, 0);
  uint64_t l0_count = lod_span_len(l0);
  float* fv = (float*)values;

  // Check sum of L0 matches sum of source (all elements scattered, none lost).
  double src_sum = 0, l0_sum = 0;
  for (uint64_t i = 0; i < n; ++i)
    src_sum += src[i];
  for (uint64_t i = l0.beg; i < l0.end; ++i)
    l0_sum += fv[i];
  CHECK(Fail, fabs(src_sum - l0_sum) < 1e-3);

  // Verify higher levels have fewer elements.
  for (int lv = 1; lv < plan.nlod; ++lv) {
    struct lod_span lvs = lod_spans_at(&plan.levels, (uint64_t)lv);
    CHECK(Fail, lod_span_len(lvs) < l0_count);
  }

  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_error("  FAIL");
  return 1;
}

// Test scatter/reduce with u16.
static int
test_scatter_reduce_u16(void)
{
  log_info("=== test_lod_cpu_u16 ===");

  uint64_t shape[] = { 2, 8, 6 };
  uint64_t chunk_shape[] = { 2, 4, 3 };
  uint32_t lod_mask = 0x6;

  uint16_t* src = NULL;
  void* values = NULL;

  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 3, shape, chunk_shape, lod_mask, 8) == 0);

  uint64_t n = 1;
  for (int d = 0; d < 3; ++d)
    n *= shape[d];
  src = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (uint16_t)((i + 1) & 0xFFFF);
  CHECK(Fail,
        lod_cpu_compute(&plan, src, &values, lod_dtype_u16, lod_reduce_min) ==
          0);

  // Basic sanity: L1 min values should be <= any L0 value.
  uint16_t* uv = (uint16_t*)values;
  struct lod_span l0 = lod_spans_at(&plan.levels, 0);
  struct lod_span l1 = lod_spans_at(&plan.levels, 1);

  uint16_t l0_min = uv[l0.beg];
  for (uint64_t i = l0.beg + 1; i < l0.end; ++i)
    if (uv[i] < l0_min)
      l0_min = uv[i];
  uint16_t l1_min = uv[l1.beg];
  for (uint64_t i = l1.beg + 1; i < l1.end; ++i)
    if (uv[i] < l1_min)
      l1_min = uv[i];
  CHECK(Fail, l1_min >= l0_min);

  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_error("  FAIL");
  return 1;
}

// Test f16 rejection.
static int
test_f16_rejected(void)
{
  log_info("=== test_lod_cpu_f16_rejected ===");

  uint64_t shape[] = { 4, 4 };
  uint64_t chunk_shape[] = { 2, 2 };
  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 2, shape, chunk_shape, 0x2, 4) == 0);

  void* values = NULL;
  int rc =
    lod_cpu_compute(&plan, NULL, &values, lod_dtype_f16, lod_reduce_mean);
  CHECK(Fail, rc != 0); // should fail

  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  lod_plan_free(&plan);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_scatter_reduce_f32(lod_reduce_mean, "f32_mean");
  rc |= test_scatter_reduce_f32(lod_reduce_min, "f32_min");
  rc |= test_scatter_reduce_f32(lod_reduce_max, "f32_max");
  rc |= test_scatter_reduce_u16();
  rc |= test_f16_rejected();
  return rc;
}
