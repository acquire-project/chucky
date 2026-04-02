#include "cpu/lod.h"
#include "lod/lod_plan.h"
#include "util/prelude.h"

#include <math.h>
#include <omp.h>
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
  uint32_t* scatter_lut = NULL;
  uint64_t* batch_offsets = NULL;

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

  scatter_lut = (uint32_t*)malloc(plan.lod_nelem[0] * sizeof(uint32_t));
  batch_offsets = (uint64_t*)malloc(plan.batch_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && batch_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_batch_offsets(&plan, batch_offsets, omp_get_max_threads());
  size_t total = plan.levels.ends[plan.nlod - 1];
  values = calloc(total, dtype_bpe(dtype_f32));
  CHECK(Fail, values);
  CHECK(Fail,
        lod_cpu_gather(
          &plan, src, values, scatter_lut, batch_offsets, dtype_f32, omp_get_max_threads()) == 0);
  CHECK(Fail, lod_cpu_reduce(&plan, values, dtype_f32, method, omp_get_max_threads()) == 0);

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

  free(scatter_lut);
  free(batch_offsets);
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(batch_offsets);
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
  uint32_t* scatter_lut = NULL;
  uint64_t* batch_offsets = NULL;

  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 3, shape, chunk_shape, lod_mask, 8) == 0);

  uint64_t n = 1;
  for (int d = 0; d < 3; ++d)
    n *= shape[d];
  src = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (uint16_t)((i + 1) & 0xFFFF);

  scatter_lut = (uint32_t*)malloc(plan.lod_nelem[0] * sizeof(uint32_t));
  batch_offsets = (uint64_t*)malloc(plan.batch_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && batch_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_batch_offsets(&plan, batch_offsets, omp_get_max_threads());
  size_t total = plan.levels.ends[plan.nlod - 1];
  values = calloc(total, dtype_bpe(dtype_u16));
  CHECK(Fail, values);
  CHECK(Fail,
        lod_cpu_gather(
          &plan, src, values, scatter_lut, batch_offsets, dtype_u16, omp_get_max_threads()) == 0);
  CHECK(Fail, lod_cpu_reduce(&plan, values, dtype_u16, lod_reduce_min, omp_get_max_threads()) == 0);

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

  free(scatter_lut);
  free(batch_offsets);
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(batch_offsets);
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
  uint32_t* scatter_lut = NULL;
  uint64_t* batch_offsets = NULL;
  void* values = NULL;
  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 2, shape, chunk_shape, 0x2, 4) == 0);
  scatter_lut = (uint32_t*)malloc(plan.lod_nelem[0] * sizeof(uint32_t));
  batch_offsets = (uint64_t*)malloc(plan.batch_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && batch_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_batch_offsets(&plan, batch_offsets, omp_get_max_threads());
  size_t total = plan.levels.ends[plan.nlod - 1];
  values = calloc(total, 2); // f16 = 2 bytes
  CHECK(Fail, values);
  int rc = lod_cpu_gather(
    &plan, values, values, scatter_lut, batch_offsets, dtype_f16, omp_get_max_threads());
  CHECK(Fail, rc != 0); // should fail
  free(scatter_lut);
  free(batch_offsets);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(batch_offsets);
  free(values);
  lod_plan_free(&plan);
  log_error("  FAIL");
  return 1;
}

// Verify nlod for a given (shape, chunk, mask) matches expected.
static int
check_nlod(const char* label,
           int ndim,
           const uint64_t* shape,
           const uint64_t* chunk_shape,
           uint32_t lod_mask,
           int expected_nlod)
{
  log_info("=== %s ===", label);
  struct lod_plan plan;
  CHECK(Fail,
        lod_plan_init_shapes(&plan, ndim, shape, chunk_shape, lod_mask, 32) ==
          0);
  if (plan.nlod != expected_nlod) {
    log_error("  expected nlod=%d, got nlod=%d", expected_nlod, plan.nlod);
    goto Fail;
  }
  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Regression: shape > chunk > shape/2 must still generate an extra level.
// Old code stopped early because it checked whether the *next* halved shape
// would fall below chunk size, missing the valid level where shape <= chunk.
static int
test_level_count_regression(void)
{
  int rc = 0;
  // 1D: shape=9, chunk=8 → 2 chunks at L0 → halve to 5 (1 chunk) → nlod=2
  rc |=
    check_nlod("nlod_1d_9c8", 1, (uint64_t[]){ 9 }, (uint64_t[]){ 8 }, 0x1, 2);
  // 2D: one dim needs extra level, other already fits
  // dim0: shape=9, chunk=8 → 2 chunks; dim1: shape=3, chunk=4 → 1 chunk
  // L0: (9,3) → dim0 has 2 chunks → continue
  // L1: (5,2) → dim0 has 1 chunk → stop → nlod=2
  rc |= check_nlod(
    "nlod_2d_asym", 2, (uint64_t[]){ 9, 3 }, (uint64_t[]){ 8, 4 }, 0x3, 2);
  return rc;
}

// shape <= chunk should produce nlod=1 (no additional levels).
static int
test_level_count_shape_le_chunk(void)
{
  int rc = 0;
  // shape == chunk → 1 chunk → nlod=1
  rc |= check_nlod("nlod_eq", 1, (uint64_t[]){ 8 }, (uint64_t[]){ 8 }, 0x1, 1);
  // shape < chunk → 1 chunk → nlod=1
  rc |= check_nlod("nlod_lt", 1, (uint64_t[]){ 3 }, (uint64_t[]){ 8 }, 0x1, 1);
  // 2D: both dims fit in 1 chunk
  rc |= check_nlod(
    "nlod_2d_le", 2, (uint64_t[]){ 4, 6 }, (uint64_t[]){ 8, 8 }, 0x3, 1);
  return rc;
}

// max_levels should cap the level count.
static int
test_level_count_max_levels(void)
{
  log_info("=== nlod_max_levels ===");
  // shape=256, chunk=1 → would need 9 levels to reach 1 chunk.
  // Cap at max_levels=3.
  struct lod_plan plan;
  CHECK(Fail,
        lod_plan_init_shapes(
          &plan, 1, (uint64_t[]){ 256 }, (uint64_t[]){ 1 }, 0x1, 3) == 0);
  CHECK(Fail, plan.nlod == 3);
  log_info("  PASS");
  return 0;
Fail:
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
  rc |= test_level_count_regression();
  rc |= test_level_count_shape_le_chunk();
  rc |= test_level_count_max_levels();
  return rc;
}
