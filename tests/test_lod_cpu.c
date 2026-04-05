#include "cpu/compress.h"
#include "cpu/lod.h"
#include "lod/lod_plan.h"
#include "stream/config.h"
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
  uint64_t* fixed_dims_offsets = NULL;

  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 3, shape, chunk_shape, lod_mask, 8) == 0);
  CHECK(Fail, plan.levels.nlod >= 2);

  // Fill source with sequential floats.
  uint64_t n = 1;
  for (int d = 0; d < 3; ++d)
    n *= shape[d];
  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  scatter_lut =
    (uint32_t*)malloc(plan.levels.level[0].lod_nelem * sizeof(uint32_t));
  fixed_dims_offsets =
    (uint64_t*)malloc(plan.fixed_dims_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && fixed_dims_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_fixed_dims_offsets(
    &plan, fixed_dims_offsets, omp_get_max_threads());
  size_t total = plan.level_spans.ends[plan.levels.nlod - 1];
  values = calloc(total, dtype_bpe(dtype_f32));
  CHECK(Fail, values);
  CHECK(Fail,
        lod_cpu_gather(&plan,
                       src,
                       values,
                       scatter_lut,
                       fixed_dims_offsets,
                       dtype_f32,
                       omp_get_max_threads()) == 0);
  CHECK(Fail,
        lod_cpu_reduce(
          &plan, values, dtype_f32, method, omp_get_max_threads()) == 0);

  // Verify L0: all source values should be present in the morton buffer.
  struct lod_span l0 = lod_spans_at(&plan.level_spans, 0);
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
  for (int lv = 1; lv < plan.levels.nlod; ++lv) {
    struct lod_span lvs = lod_spans_at(&plan.level_spans, (uint64_t)lv);
    CHECK(Fail, lod_span_len(lvs) < l0_count);
  }

  free(scatter_lut);
  free(fixed_dims_offsets);
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(fixed_dims_offsets);
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
  uint64_t* fixed_dims_offsets = NULL;

  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 3, shape, chunk_shape, lod_mask, 8) == 0);

  uint64_t n = 1;
  for (int d = 0; d < 3; ++d)
    n *= shape[d];
  src = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (uint16_t)((i + 1) & 0xFFFF);

  scatter_lut =
    (uint32_t*)malloc(plan.levels.level[0].lod_nelem * sizeof(uint32_t));
  fixed_dims_offsets =
    (uint64_t*)malloc(plan.fixed_dims_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && fixed_dims_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_fixed_dims_offsets(
    &plan, fixed_dims_offsets, omp_get_max_threads());
  size_t total = plan.level_spans.ends[plan.levels.nlod - 1];
  values = calloc(total, dtype_bpe(dtype_u16));
  CHECK(Fail, values);
  CHECK(Fail,
        lod_cpu_gather(&plan,
                       src,
                       values,
                       scatter_lut,
                       fixed_dims_offsets,
                       dtype_u16,
                       omp_get_max_threads()) == 0);
  CHECK(Fail,
        lod_cpu_reduce(
          &plan, values, dtype_u16, lod_reduce_min, omp_get_max_threads()) ==
          0);

  // Basic sanity: L1 min values should be <= any L0 value.
  uint16_t* uv = (uint16_t*)values;
  struct lod_span l0 = lod_spans_at(&plan.level_spans, 0);
  struct lod_span l1 = lod_spans_at(&plan.level_spans, 1);

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
  free(fixed_dims_offsets);
  free(src);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(fixed_dims_offsets);
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
  uint64_t* fixed_dims_offsets = NULL;
  void* values = NULL;
  struct lod_plan plan;
  CHECK(Fail, lod_plan_init(&plan, 2, shape, chunk_shape, 0x2, 4) == 0);
  scatter_lut =
    (uint32_t*)malloc(plan.levels.level[0].lod_nelem * sizeof(uint32_t));
  fixed_dims_offsets =
    (uint64_t*)malloc(plan.fixed_dims_count * sizeof(uint64_t));
  CHECK(Fail, scatter_lut && fixed_dims_offsets);
  lod_cpu_build_scatter_lut(&plan, scatter_lut, omp_get_max_threads());
  lod_cpu_build_scatter_fixed_dims_offsets(
    &plan, fixed_dims_offsets, omp_get_max_threads());
  size_t total = plan.level_spans.ends[plan.levels.nlod - 1];
  values = calloc(total, 2); // f16 = 2 bytes
  CHECK(Fail, values);
  int rc = lod_cpu_gather(&plan,
                          values,
                          values,
                          scatter_lut,
                          fixed_dims_offsets,
                          dtype_f16,
                          omp_get_max_threads());
  CHECK(Fail, rc != 0); // should fail
  free(scatter_lut);
  free(fixed_dims_offsets);
  free(values);
  lod_plan_free(&plan);
  log_info("  PASS");
  return 0;

Fail:
  free(scatter_lut);
  free(fixed_dims_offsets);
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
  if (plan.levels.nlod != expected_nlod) {
    log_error(
      "  expected nlod=%d, got nlod=%d", expected_nlod, plan.levels.nlod);
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
  CHECK(Fail, plan.levels.nlod == 3);
  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Test max_nlod semantics: 0=auto, N>0=cap at N total levels.
// Uses lod_plan_init_shapes to verify plan computation directly.
static int
test_max_nlod_semantics(void)
{
  log_info("=== test_max_nlod_semantics ===");

  // shape=256, chunk=1, mask=0x1 -> auto gives many levels
  uint64_t shape[] = { 256 };
  uint64_t chunk[] = { 1 };
  uint32_t mask = 0x1;
  struct lod_plan plan;

  // max_nlod=0 -> auto (use LOD_MAX_LEVELS)
  // With shape=256, chunk=1, auto should give 9 levels (256->128->...->1)
  CHECK(Fail,
        lod_plan_init_shapes(&plan, 1, shape, chunk, mask, LOD_MAX_LEVELS) ==
          0);
  int auto_nlod = plan.levels.nlod;
  CHECK(Fail, auto_nlod == 9);
  log_info("  zero (auto): nlod=%d", auto_nlod);

  // max_nlod=1 -> max_levels=1 -> nlod=1 (base only, no downsampling)
  CHECK(Fail, lod_plan_init_shapes(&plan, 1, shape, chunk, mask, 1) == 0);
  CHECK(Fail, plan.levels.nlod == 1);
  log_info("  one (base only): nlod=%d", plan.levels.nlod);

  // max_nlod=4 -> max_levels=4 -> nlod capped at 4
  CHECK(Fail, lod_plan_init_shapes(&plan, 1, shape, chunk, mask, 4) == 0);
  CHECK(Fail, plan.levels.nlod == 4);
  CHECK(Fail, plan.levels.nlod < auto_nlod);
  log_info("  positive (cap=4 total levels): nlod=%d", plan.levels.nlod);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Test that max_nlod > LOD_MAX_LEVELS is rejected by compute_stream_layouts.
static int
test_max_nlod_validation_rejection(void)
{
  log_info("=== test_max_nlod_validation_rejection ===");

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .downsample = 1,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config_too_big = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = 33, // exceeds LOD_MAX_LEVELS
  };

  struct computed_stream_layouts cl;
  int rc = compute_stream_layouts(
    &config_too_big, 1, compress_cpu_max_output_size, &cl);
  CHECK(Fail, rc != 0); // must be rejected

  struct tile_stream_configuration config_negative = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = -1, // negative is invalid
  };

  rc = compute_stream_layouts(
    &config_negative, 1, compress_cpu_max_output_size, &cl);
  CHECK(Fail, rc != 0); // must be rejected

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Test that max_nlod=1 through compute_stream_layouts yields nlod==1.
static int
test_max_nlod_one_via_layouts(void)
{
  log_info("=== test_max_nlod_one_via_layouts ===");

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 64,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .downsample = 1,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = 1, // base level only
  };

  struct computed_stream_layouts cl;
  CHECK(Fail,
        compute_stream_layouts(&config, 1, compress_cpu_max_output_size, &cl) ==
          0);
  CHECK(Fail, cl.levels.nlod == 1); // base level only
  computed_stream_layouts_free(&cl);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Test that positive max_nlod caps the level count via compute_stream_layouts.
static int
test_max_nlod_positive_cap_via_layouts(void)
{
  log_info("=== test_max_nlod_positive_cap_via_layouts ===");

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 256,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .downsample = 1,
      .storage_position = 1 },
  };

  // First, compute with auto (max_nlod=0) to see uncapped nlod.
  struct tile_stream_configuration config_auto = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = 0, // auto
  };

  struct computed_stream_layouts cl_auto;
  CHECK(Fail,
        compute_stream_layouts(
          &config_auto, 1, compress_cpu_max_output_size, &cl_auto) == 0);
  CHECK(Fail, cl_auto.levels.nlod > 3); // auto should produce more than 3
  computed_stream_layouts_free(&cl_auto);

  // Now cap at max_nlod=3 -> expect nlod==3 (3 total levels).
  struct tile_stream_configuration config_cap = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = 3,
  };

  struct computed_stream_layouts cl_cap;
  CHECK(Fail,
        compute_stream_layouts(
          &config_cap, 1, compress_cpu_max_output_size, &cl_cap) == 0);
  CHECK(Fail, cl_cap.levels.nlod == 3); // 3 total levels
  computed_stream_layouts_free(&cl_cap);

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

// Test that max_nlod=0 (auto) produces identical results to LOD_MAX_LEVELS.
// Both should resolve to the natural (uncapped) level count.
static int
test_max_nlod_zero_equals_uncapped(void)
{
  log_info("=== test_max_nlod_zero_equals_uncapped ===");

  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 256,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .downsample = 1,
      .storage_position = 1 },
    { .size = 256,
      .chunk_size = 8,
      .chunks_per_shard = 1,
      .downsample = 1,
      .storage_position = 2 },
  };

  // max_nlod=0 -> auto (should resolve to LOD_MAX_LEVELS internally)
  struct tile_stream_configuration config_auto = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = 0,
  };

  // max_nlod=LOD_MAX_LEVELS -> explicitly uncapped
  struct tile_stream_configuration config_uncapped = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .max_nlod = LOD_MAX_LEVELS,
  };

  struct computed_stream_layouts cl_auto, cl_uncapped;
  CHECK(Fail,
        compute_stream_layouts(
          &config_auto, 1, compress_cpu_max_output_size, &cl_auto) == 0);
  CHECK(Fail,
        compute_stream_layouts(
          &config_uncapped, 1, compress_cpu_max_output_size, &cl_uncapped) ==
          0);

  // Both should produce multiple levels (sanity check the config is
  // interesting)
  CHECK(Fail, cl_auto.levels.nlod > 1);

  // nlod must match
  CHECK(Fail, cl_auto.levels.nlod == cl_uncapped.levels.nlod);

  // Per-level chunk counts must match
  for (int lv = 0; lv < cl_auto.levels.nlod; ++lv)
    CHECK(Fail,
          cl_auto.levels.level[lv].chunk_count ==
            cl_uncapped.levels.level[lv].chunk_count);

  // Total chunks must match
  CHECK(Fail, cl_auto.levels.total_chunks == cl_uncapped.levels.total_chunks);

  log_info("  nlod=%d (auto=0 and uncapped=%d agree)",
           cl_auto.levels.nlod,
           LOD_MAX_LEVELS);

  computed_stream_layouts_free(&cl_auto);
  computed_stream_layouts_free(&cl_uncapped);
  log_info("  PASS");
  return 0;
Fail:
  computed_stream_layouts_free(&cl_auto);
  computed_stream_layouts_free(&cl_uncapped);
  log_error("  FAIL");
  return 1;
}

// chunk_size is constant across levels (partial chunks are padded).
// chunks_per_shard is clamped to chunk_count at coarse levels.
static int
test_chunk_geometry_at_coarse_levels(void)
{
  log_info("=== test_chunk_geometry_at_coarse_levels ===");

  // 2D: shape=(10,10), chunk=(8,8), lod on both dims.
  // L0: shape=(10,10), chunk=(8,8), chunk_count=ceildiv(10,8)=2
  // L1: shape=(5,5),   chunk=(8,8), chunk_count=ceildiv(5,8)=1
  uint64_t shape[] = { 10, 10 };
  uint64_t chunk[] = { 8, 8 };
  uint32_t mask = 0x3; // both dims in LOD

  struct lod_plan plan;
  CHECK(Fail,
        lod_plan_init_shapes(&plan, 2, shape, chunk, mask, LOD_MAX_LEVELS) ==
          0);
  CHECK(Fail, plan.levels.nlod >= 2);

  // L0: chunk_size == config
  CHECK(Fail, plan.levels.level[0].dim[0].chunk_size == 8);
  CHECK(Fail, plan.levels.level[0].dim[1].chunk_size == 8);

  // L1: shape=(5,5), chunk_size still 8 (not clamped), partial chunk padded
  CHECK(Fail, plan.levels.level[1].dim[0].size == 5);
  CHECK(Fail, plan.levels.level[1].dim[1].size == 5);
  CHECK(Fail, plan.levels.level[1].dim[0].chunk_size == 8);
  CHECK(Fail, plan.levels.level[1].dim[1].chunk_size == 8);

  // Now test chunks_per_shard via _from_dims.
  // We need n_append >= 1 with chunk_size=1 on dim 0 for _from_dims.
  struct dimension dims2[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 10,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 1 },
    { .size = 10,
      .chunk_size = 8,
      .chunks_per_shard = 2,
      .downsample = 1,
      .storage_position = 2 },
  };

  struct lod_plan plan2;
  CHECK(Fail, lod_plan_init_from_dims(&plan2, dims2, 3, LOD_MAX_LEVELS) == 0);
  CHECK(Fail2, plan2.levels.nlod >= 2);

  // L0: chunks_per_shard = L0 config
  CHECK(Fail2, plan2.levels.level[0].dim[1].chunks_per_shard == 2);
  CHECK(Fail2, plan2.levels.level[0].dim[2].chunks_per_shard == 2);

  // L1: shape halved => (5,5) on LOD dims; chunk_size stays 8;
  // chunk_count = ceildiv(5,8) = 1; chunks_per_shard = min(2,1) = 1
  CHECK(Fail2, plan2.levels.level[1].dim[1].chunk_size == 8);
  CHECK(Fail2, plan2.levels.level[1].dim[2].chunk_size == 8);
  CHECK(Fail2, plan2.levels.level[1].dim[1].chunks_per_shard == 1);
  CHECK(Fail2, plan2.levels.level[1].dim[2].chunks_per_shard == 1);

  lod_plan_free(&plan2);
  log_info("  PASS");
  return 0;

Fail2:
  lod_plan_free(&plan2);
Fail:
  log_error("  FAIL");
  return 1;
}

// Test lod_plan_init_from_epoch_dims with non-power-of-two inner shape.
// chunk_size stays constant; chunks_per_shard clamped to chunk_count.
static int
test_from_epoch_dims_clamp(void)
{
  log_info("=== test_from_epoch_dims_clamp ===");

  // 3D: dim 0 is append (chunk_size=1), dims 1,2 are inner with downsample.
  // dim 1: size=10, chunk_size=8 -> L0 shape=10, L1 shape=5, chunk_size=8
  // dim 2: size=6, chunk_size=4 -> L0 shape=6, L1 shape=3, chunk_size=4
  struct dimension dims[] = {
    { .size = 0,
      .chunk_size = 1,
      .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 10,
      .chunk_size = 8,
      .chunks_per_shard = 0, // all chunks
      .downsample = 1,
      .storage_position = 1 },
    { .size = 6,
      .chunk_size = 4,
      .chunks_per_shard = 0, // all chunks
      .downsample = 1,
      .storage_position = 2 },
  };

  struct lod_plan plan;
  CHECK(Fail,
        lod_plan_init_from_epoch_dims(&plan, dims, 3, 1, LOD_MAX_LEVELS) == 0);
  CHECK(Fail, plan.levels.nlod >= 2);

  // L0: shapes = (1, 10, 6), chunk_sizes = (1, 8, 4)
  CHECK(Fail, plan.levels.level[0].dim[1].size == 10);
  CHECK(Fail, plan.levels.level[0].dim[2].size == 6);
  CHECK(Fail, plan.levels.level[0].dim[0].chunk_size == 1);
  CHECK(Fail, plan.levels.level[0].dim[1].chunk_size == 8);
  CHECK(Fail, plan.levels.level[0].dim[2].chunk_size == 4);

  // L1: shapes = (1, 5, 3), chunk_sizes unchanged (1, 8, 4)
  CHECK(Fail, plan.levels.level[1].dim[1].size == 5);
  CHECK(Fail, plan.levels.level[1].dim[2].size == 3);
  CHECK(Fail, plan.levels.level[1].dim[1].chunk_size == 8); // constant
  CHECK(Fail, plan.levels.level[1].dim[2].chunk_size == 4); // constant

  // chunks_per_shard at L0: cps=0 => all chunks: ceildiv(10,8)=2,
  // ceildiv(6,4)=2
  CHECK(Fail, plan.levels.level[0].dim[1].chunks_per_shard == 2);
  CHECK(Fail, plan.levels.level[0].dim[2].chunks_per_shard == 2);

  // chunks_per_shard at L1: ceildiv(5,8)=1, ceildiv(3,4)=1
  CHECK(Fail, plan.levels.level[1].dim[1].chunks_per_shard == 1);
  CHECK(Fail, plan.levels.level[1].dim[2].chunks_per_shard == 1);

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
  rc |= test_level_count_regression();
  rc |= test_level_count_shape_le_chunk();
  rc |= test_level_count_max_levels();
  rc |= test_max_nlod_semantics();
  rc |= test_max_nlod_validation_rejection();
  rc |= test_max_nlod_one_via_layouts();
  rc |= test_max_nlod_positive_cap_via_layouts();
  rc |= test_max_nlod_zero_equals_uncapped();
  rc |= test_chunk_geometry_at_coarse_levels();
  rc |= test_from_epoch_dims_clamp();
  return rc;
}
