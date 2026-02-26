#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM 64

static uint64_t
max_shape(int ndim, const uint64_t* shape)
{
  uint64_t m = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > m)
      m = shape[d];
  return m;
}

// Smallest p such that 2^p >= v. Returns 0 for v <= 1.
static int
ceil_log2(uint64_t v)
{
  int p = 0;
  while ((1ull << p) < v)
    ++p;
  return p;
}

// Clamped extent: how many valid coordinates in [lo, lo+scale) for a
// dimension of size `shape_d`.
static uint64_t
clamped_extent(uint64_t shape_d, uint64_t lo, uint64_t scale)
{
  if (lo >= shape_d)
    return 0;
  uint64_t e = shape_d - lo;
  return (e < scale) ? e : scale;
}

// Count how many Morton codes in [0, k) decode to coordinates within the array
// bounds shape[0..ndim-1].
//
// coords: the per-dimension coordinates identifying the Morton code.
// depth:  number of extra zero-digit levels appended below coords (handles
//         the equivalent of shifting a Morton code left by depth*ndim bits).
//
// Walks the digit sequence top-down, one ndim-bit digit per level.
// At each level, sibling subtrees before the target digit contribute a known
// count. Instead of enumerating all v < digit (2^d siblings), we decompose the
// digit bit by bit across dimensions. For each dimension where the digit's bit
// is 1, we split: that dimension takes 0 (all lower dimensions free) or takes 1
// (continue matching). This is O(d) per level, or O(p*d) overall.
static uint64_t
morton_rank(int ndim,
            const uint64_t* shape,
            const uint64_t* coords,
            int depth)
{
  int p = ceil_log2(max_shape(ndim, shape));

  // Ensure p covers all coordinate bits
  for (int d = 0; d < ndim; ++d) {
    int pc = coords[d] > 0 ? ceil_log2(coords[d] + 1) : 0;
    if (pc > p)
      p = pc;
  }

  // Total digit levels = coordinate bits + extra zero levels
  int total_levels = p + depth;

  uint64_t count = 0;
  uint64_t prefix[MAX_NDIM] = { 0 };

  for (int level = 0; level < total_levels; ++level) {
    uint64_t scale = 1ull << (total_levels - 1 - level);

    // Extract digit: from coords for levels 0..p-1, zero for deeper levels
    int digit = 0;
    if (level < p) {
      int bit_idx = p - 1 - level;
      for (int d = 0; d < ndim; ++d)
        digit |= (int)((coords[d] >> bit_idx) & 1) << d;
    }

    // For each dimension, precompute clamped extents for bit=0 and bit=1.
    uint64_t ext[MAX_NDIM][2];
    for (int d = 0; d < ndim; ++d) {
      for (int b = 0; b < 2; ++b) {
        uint64_t lo = (prefix[d] * 2 + (uint64_t)b) * scale;
        ext[d][b] = clamped_extent(shape[d], lo, scale);
      }
    }

    // Prefix product of "free" extents (both bit values) for dims 0..d-1.
    uint64_t free_prefix[MAX_NDIM + 1];
    free_prefix[0] = 1;
    for (int d = 0; d < ndim; ++d)
      free_prefix[d + 1] = free_prefix[d] * (ext[d][0] + ext[d][1]);

    // Scan dimensions from highest to lowest. For each dimension where
    // the digit's bit is 1, split: this dim takes 0 (lower dims free),
    // or this dim takes 1 (continue matching tight prefix above).
    uint64_t tight = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      int bit = (digit >> d) & 1;
      if (bit == 1)
        count += tight * ext[d][0] * free_prefix[d];
      tight *= ext[d][bit];
    }

    // Descend: extend prefix with this digit's per-dimension bits
    for (int d = 0; d < ndim; ++d)
      prefix[d] = prefix[d] * 2 + ((digit >> d) & 1);
  }

  return count;
}

static int
is_all_ones(int n, const uint64_t* v)
{
  for (int d = 0; d < n; ++d)
    if (v[d] > 1)
      return 0;
  return 1;
}

// Row-major linear index to coordinates.
static void
linear_to_coords(int ndim,
                 const uint64_t* shape,
                 uint64_t idx,
                 uint64_t* coords)
{
  for (int d = 0; d < ndim; ++d) {
    coords[d] = idx % shape[d];
    idx /= shape[d];
  }
}

#define MAX_LOD 32

struct slice
{
  uint64_t beg, end;
};

static uint64_t
slice_len(struct slice s)
{
  return s.end - s.beg;
}

struct spans
{
  uint64_t* ends; // ends[i] = exclusive end of span i
  uint64_t n;     // number of spans
};

static struct slice
spans_at(const struct spans* s, uint64_t i)
{
  return (struct slice){
    .beg = i > 0 ? s->ends[i - 1] : 0,
    .end = s->ends[i],
  };
}

struct lod_plan
{
  int ndim; // full dimensionality
  int nlev;
  uint64_t shapes[MAX_LOD][MAX_NDIM]; // full shapes per level

  // ds/batch decomposition
  uint8_t ds_mask; // which dims are downsampled (bit d set => dim d halved)
  int ds_ndim;
  int ds_map[MAX_NDIM];    // ds_map[k] = full dim index for ds dim k
  int batch_ndim;
  int batch_map[MAX_NDIM]; // batch_map[k] = full dim index for batch dim k
  uint64_t batch_shape[MAX_NDIM];
  uint64_t batch_count;

  uint64_t ds_shapes[MAX_LOD][MAX_NDIM]; // ds-only shapes per level
  uint64_t ds_counts[MAX_LOD];           // product of ds_shapes per level

  struct spans levels;    // full level ends in values buf (batch_count * ds cumulative)
  struct spans ds_levels; // ds-only level ends (one batch slice)
  uint64_t* ends;         // child-group segment ends, relative within batch slice
};

// Segment slice for level l's child-group ends in the ends buffer.
// Uses ds_levels since ends are relative within one batch slice.
static struct slice
lod_segment(const struct lod_plan* p, int level)
{
  struct slice next = spans_at(&p->ds_levels, level + 1);
  uint64_t base = p->ds_levels.ends[0]; // = ds_counts[0]
  return (struct slice){ .beg = next.beg - base, .end = next.end - base };
}

static void
lod_plan_free(struct lod_plan* p);

// Increment coordinates to the next position in Morton order.
// This is equivalent to adding 1 to the Morton code: carry-propagating
// addition on the mixed-radix digit array (each digit base 2).
// On overflow, sets coords[0] = 2^p (one past the valid range) so that
// morton_rank with these coords correctly counts all valid elements.
static void
coords_morton_next(int ndim, int p, uint64_t* coords)
{
  for (int bit = 0; bit < p; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      uint64_t mask = 1ull << bit;
      coords[d] ^= mask;
      if (coords[d] & mask)
        return; // no carry
    }
  }
  // overflow
  memset(coords, 0, (size_t)ndim * sizeof(uint64_t));
  coords[0] = 1ull << p;
}

// Fill one level's segment-end array. Each iteration
// is independent (one iteration could be one GPU thread).
//
// ends[pos] receives the exclusive end of parent pos's children in the
// contiguous values buffer (relative within one batch slice).
static void
lod_fill_ends(int ndim,
              const uint64_t* child_shape,
              const uint64_t* parent_shape,
              uint64_t n_parents,
              uint64_t* ends)
{
  int p = ceil_log2(max_shape(ndim, parent_shape));

  uint64_t coords[MAX_NDIM];
  uint64_t next[MAX_NDIM];
  for (uint64_t j = 0; j < n_parents; ++j) {
    linear_to_coords(ndim, parent_shape, j, coords);
    uint64_t pos = morton_rank(ndim, parent_shape, coords, 0);

    // (m+1) << ndim: next parent's coords at child level with depth=1
    memcpy(next, coords, (size_t)ndim * sizeof(uint64_t));
    coords_morton_next(ndim, p, next);
    uint64_t val = morton_rank(ndim, child_shape, next, 1);

    ends[pos] = val;
  }
}

// Compute batch linear index from full coordinates.
static uint64_t
plan_batch_index(const struct lod_plan* p, const uint64_t* full_coords)
{
  uint64_t idx = 0, stride = 1;
  for (int k = 0; k < p->batch_ndim; ++k) {
    idx += full_coords[p->batch_map[k]] * stride;
    stride *= p->batch_shape[k];
  }
  return idx;
}

// Extract ds coordinates from full coordinates.
static void
plan_extract_ds(const struct lod_plan* p,
                const uint64_t* full_coords,
                uint64_t* ds_coords)
{
  for (int k = 0; k < p->ds_ndim; ++k)
    ds_coords[k] = full_coords[p->ds_map[k]];
}

// Compute the shape pyramid, buffer layout, and child-group segment ends.
// ds_mask: bit d set means dimension d is downsampled (halved each level).
// If ds_mask has all bits set for dims 0..ndim-1, all dims are downsampled.
// If ds_mask is 0, no downsampling: nlev=1.
// Returns 0 on failure.
static int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              uint8_t ds_mask,
              int max_levels)
{
  memset(p, 0, sizeof(*p));
  p->ndim = ndim;
  p->ds_mask = ds_mask;

  // Build ds/batch dimension maps
  for (int d = 0; d < ndim; ++d) {
    if (ds_mask & (1 << d)) {
      p->ds_map[p->ds_ndim++] = d;
    } else {
      p->batch_map[p->batch_ndim] = d;
      p->batch_shape[p->batch_ndim] = shape[d];
      p->batch_ndim++;
    }
  }
  p->batch_count = 1;
  for (int k = 0; k < p->batch_ndim; ++k)
    p->batch_count *= p->batch_shape[k];

  // Initialize full and ds shapes at level 0
  memcpy(p->shapes[0], shape, (size_t)ndim * sizeof(uint64_t));
  for (int k = 0; k < p->ds_ndim; ++k)
    p->ds_shapes[0][k] = shape[p->ds_map[k]];

  // Build shape pyramid: only halve ds dims
  p->nlev = 1;
  while (p->nlev < max_levels &&
         !is_all_ones(p->ds_ndim, p->ds_shapes[p->nlev - 1])) {
    // ds dims: halve
    for (int k = 0; k < p->ds_ndim; ++k)
      p->ds_shapes[p->nlev][k] = (p->ds_shapes[p->nlev - 1][k] + 1) / 2;
    // full shapes: copy all, then overwrite ds dims
    memcpy(p->shapes[p->nlev], p->shapes[p->nlev - 1],
           (size_t)ndim * sizeof(uint64_t));
    for (int k = 0; k < p->ds_ndim; ++k)
      p->shapes[p->nlev][p->ds_map[k]] = p->ds_shapes[p->nlev][k];
    ++p->nlev;
  }

  // ds element counts per level
  for (int k = 0; k < p->nlev; ++k) {
    p->ds_counts[k] = 1;
    for (int d = 0; d < p->ds_ndim; ++d)
      p->ds_counts[k] *= p->ds_shapes[k][d];
  }

  // ds_levels: per-batch-slice level ends
  p->ds_levels.n = (uint64_t)p->nlev;
  p->ds_levels.ends = (uint64_t*)malloc(p->nlev * sizeof(uint64_t));
  if (!p->ds_levels.ends)
    goto Fail;
  p->ds_levels.ends[0] = p->ds_counts[0];
  for (int k = 1; k < p->nlev; ++k)
    p->ds_levels.ends[k] = p->ds_levels.ends[k - 1] + p->ds_counts[k];

  // levels: full (batch-multiplied) level ends
  p->levels.n = (uint64_t)p->nlev;
  p->levels.ends = (uint64_t*)malloc(p->nlev * sizeof(uint64_t));
  if (!p->levels.ends)
    goto Fail;
  for (int k = 0; k < p->nlev; ++k)
    p->levels.ends[k] = p->batch_count * p->ds_levels.ends[k];

  // Compute child-group segment ends for all ds-level transitions.
  // Total ends = sum(ds_counts[1..nlev-1]).
  // Ends are relative within one batch slice.
  {
    uint64_t total_ends =
      p->ds_levels.ends[p->nlev - 1] - p->ds_levels.ends[0];
    if (total_ends > 0) {
      p->ends = (uint64_t*)malloc(total_ends * sizeof(uint64_t));
      if (!p->ends)
        goto Fail;
      for (int l = 0; l < p->nlev - 1; ++l) {
        struct slice seg = lod_segment(p, l);
        lod_fill_ends(p->ds_ndim,
                      p->ds_shapes[l],
                      p->ds_shapes[l + 1],
                      slice_len(seg),
                      p->ends + seg.beg);
      }
    }
  }

  return 1;
Fail:
  lod_plan_free(p);
  return 0;
}

static void
lod_plan_free(struct lod_plan* p)
{
  if (!p)
    return;
  free(p->levels.ends);
  free(p->ds_levels.ends);
  free(p->ends);
  memset(p, 0, sizeof(*p));
}

// Scatter array (row-major) into compacted Morton order for level 0.
// Batch dimensions provide a flat outer index; Morton ordering applies
// only to ds dimensions.
static void
lod_scatter(const struct lod_plan* p, const float* src, float* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = slice_len(spans_at(&p->levels, 0));

  uint64_t full_coords[MAX_NDIM];
  uint64_t ds_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_ds(p, full_coords, ds_coords);
    uint64_t pos = morton_rank(p->ds_ndim, p->ds_shapes[0], ds_coords, 0);
    dst[b * p->ds_counts[0] + pos] = src[i];
  }
}

// Segmented mean reduction over all LOD levels.
// Processes each level transition and each batch slice independently.
// The ends array is relative within one batch slice.
static void
lod_reduce(const struct lod_plan* p, float* values)
{
  for (int l = 0; l < p->nlev - 1; ++l) {
    struct slice seg = lod_segment(p, l);
    uint64_t src_ds = p->ds_counts[l];
    uint64_t dst_ds = p->ds_counts[l + 1];
    struct slice src_level = spans_at(&p->levels, l);
    struct slice dst_level = spans_at(&p->levels, l + 1);

    for (uint64_t b = 0; b < p->batch_count; ++b) {
      uint64_t src_base = src_level.beg + b * src_ds;
      uint64_t dst_base = dst_level.beg + b * dst_ds;

      for (uint64_t i = 0; i < dst_ds; ++i) {
        uint64_t start = (i > 0) ? p->ends[seg.beg + i - 1] : 0;
        uint64_t end = p->ends[seg.beg + i];
        uint64_t len = end - start;
        float sum = 0;
        for (uint64_t j = start; j < end; ++j)
          sum += values[src_base + j];
        values[dst_base + i] = sum / (float)len;
      }
    }
  }
}

// Compute all LOD levels into a contiguous values buffer.
// *out_values: contiguous buffer laid out as:
//   [level 0: batch_count * ds_counts[0]] [level 1: ...] ...
// Within each level, batch slices are contiguous, each in compacted Morton
// order over the ds dimensions.
// Returns 0 on failure. Caller must free out_values.
static int
lod_compute(const struct lod_plan* p, const float* src, float** out_values)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlev - 1];
  float* values = (float*)malloc(total_vals * sizeof(float));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter(p, src, values);
  lod_reduce(p, values);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}

// Reference: brute-force downsample by averaging only valid children.
// ds_mask controls which dimensions are halved; non-masked dimensions
// keep the same coordinate (no 2x expansion).
static void
downsample_ref(int ndim,
               uint8_t ds_mask,
               const uint64_t* cur_shape,
               const uint64_t* next_shape,
               const float* src,
               float* dst)
{
  uint64_t n_next = 1;
  for (int d = 0; d < ndim; ++d)
    n_next *= next_shape[d];

  int n_ds = 0;
  for (int d = 0; d < ndim; ++d)
    if (ds_mask & (1 << d))
      n_ds++;
  int n_children = 1 << n_ds;

  uint64_t coords[MAX_NDIM];
  for (uint64_t j = 0; j < n_next; ++j) {
    linear_to_coords(ndim, next_shape, j, coords);

    float sum = 0;
    int count = 0;
    for (int c = 0; c < n_children; ++c) {
      uint64_t lin = 0;
      uint64_t stride = 1;
      int valid = 1;
      int ds_bit = 0;

      for (int d = 0; d < ndim; ++d) {
        uint64_t child_coord;
        if (ds_mask & (1 << d)) {
          child_coord = coords[d] * 2 + ((c >> ds_bit) & 1);
          ds_bit++;
        } else {
          child_coord = coords[d];
        }
        if (child_coord >= cur_shape[d]) {
          valid = 0;
          break;
        }
        lin += child_coord * stride;
        stride *= cur_shape[d];
      }
      if (valid) {
        sum += src[lin];
        ++count;
      }
    }
    dst[j] = sum / (float)count;
  }
}

// Unshuffle: convert from compacted Morton order (with batch dims) back to
// row-major at the given level.
static void
morton_unshuffle(const struct lod_plan* p,
                 int level,
                 const float* morton_buf,
                 float* rowmajor)
{
  const uint64_t* full_shape = p->shapes[level];
  uint64_t n = 1;
  for (int d = 0; d < p->ndim; ++d)
    n *= full_shape[d];

  uint64_t full_coords[MAX_NDIM];
  uint64_t ds_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_ds(p, full_coords, ds_coords);
    uint64_t pos =
      morton_rank(p->ds_ndim, p->ds_shapes[level], ds_coords, 0);
    rowmajor[i] = morton_buf[b * p->ds_counts[level] + pos];
  }
}

// --- Reference implementations for testing (use old uint64_t Morton codes) ---

// Decode a Morton code back into coordinates (reference, limited to small
// shapes).
static void
morton_decode(int ndim, uint64_t code, uint64_t* coords)
{
  memset(coords, 0, (size_t)ndim * sizeof(*coords));
  for (int bit = 0; bit < 64 / ndim; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      coords[d] |= (code & 1) << bit;
      code >>= 1;
    }
  }
}

// brute-force rank using Morton codes (reference)
static uint64_t
morton_rank_ref(int ndim, const uint64_t* shape, uint64_t k)
{
  uint64_t count = 0;
  uint64_t coords[MAX_NDIM];
  for (uint64_t m = 0; m < k; ++m) {
    morton_decode(ndim, m, coords);
    int valid = 1;
    for (int d = 0; d < ndim; ++d) {
      if (coords[d] >= shape[d]) {
        valid = 0;
        break;
      }
    }
    count += valid;
  }
  return count;
}

static int
test_3d(void)
{
  printf("--- test_3d ---\n");
  int ok = 0;
  const int ndim = 3;
  const uint64_t shape[] = { 3, 2, 5 };

  int p = ceil_log2(max_shape(ndim, shape));
  uint64_t box = 1ull << (ndim * p);
  for (uint64_t k = 0; k <= box; ++k) {
    // Decode k to coords, then use new morton_rank
    uint64_t coords[MAX_NDIM];
    morton_decode(ndim, k, coords);
    uint64_t r = morton_rank(ndim, shape, coords, 0);

    // But we need to compare against the reference which counts codes in [0,k).
    // morton_rank(shape, coords, 0) counts codes < encode(coords), not codes < k.
    // So compare against brute-force directly.
    uint64_t r_ref = morton_rank_ref(ndim, shape, k);
    if (r != r_ref) {
      printf("  FAIL at k=%llu: got %llu, expected %llu\n",
             (unsigned long long)k,
             (unsigned long long)r,
             (unsigned long long)r_ref);
      goto Fail;
    }
  }
  uint64_t total = morton_rank_ref(ndim, shape, 1ull << (ndim * p));
  printf("  total valid in 3x2x5 = %llu\n", (unsigned long long)total);
  CHECK(Fail, total == 30);
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_1d(void)
{
  printf("--- test_1d ---\n");
  int ok = 0;
  const int ndim = 1;
  const uint64_t shape[] = { 7 };
  for (uint64_t k = 0; k <= 8; ++k) {
    uint64_t coords[MAX_NDIM];
    morton_decode(ndim, k, coords);
    uint64_t r = morton_rank(ndim, shape, coords, 0);
    uint64_t r_ref = morton_rank_ref(ndim, shape, k);
    CHECK(Fail, r == r_ref);
  }
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_lod(const char* label,
         int ndim,
         const uint64_t* shape,
         uint8_t ds_mask)
{
  printf("--- %s ---\n", label);
  int ok = 0;
  float* src = NULL;
  float* values = NULL;
  struct lod_plan plan = { 0 };
  float* prev_rm = NULL;
  float* ref = NULL;
  float* cur_rm = NULL;

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, ds_mask, MAX_LOD));
  printf("  ds_mask=0x%x  ds_ndim=%d  batch_ndim=%d  batch_count=%llu\n",
         ds_mask, plan.ds_ndim, plan.batch_ndim,
         (unsigned long long)plan.batch_count);

  CHECK(Fail, lod_compute(&plan, src, &values));
  printf("  levels: %d\n", plan.nlev);

  if (plan.nlev < 2) {
    // No downsampling levels — just verify scatter roundtrip
    prev_rm = (float*)malloc(n * sizeof(float));
    CHECK(Fail, prev_rm);
    morton_unshuffle(&plan, 0, values, prev_rm);
    for (uint64_t i = 0; i < n; ++i) {
      if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
        printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
               (unsigned long long)i, prev_rm[i], src[i]);
        goto Fail;
      }
    }
    printf("  level 0 scatter: ok (no downsample levels)\n");
    printf("  PASS\n");
    ok = 1;
    goto Fail; // cleanup
  }

  CHECK(Fail, plan.nlev >= 2);

  // Verify level 0 scatter roundtrips
  prev_rm = (float*)malloc(n * sizeof(float));
  CHECK(Fail, prev_rm);
  morton_unshuffle(&plan, 0, values, prev_rm);
  for (uint64_t i = 0; i < n; ++i) {
    if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
      printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
             (unsigned long long)i, prev_rm[i], src[i]);
      goto Fail;
    }
  }
  printf("  level 0 scatter: ok\n");

  // Verify each subsequent level against brute-force downsample
  for (int l = 1; l < plan.nlev; ++l) {
    const uint64_t* prev_shape = plan.shapes[l - 1];
    const uint64_t* cur_shape = plan.shapes[l];
    struct slice lev = spans_at(&plan.levels, l);
    uint64_t cur_n = slice_len(lev);

    ref = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, ref);
    downsample_ref(ndim, ds_mask, prev_shape, cur_shape, prev_rm, ref);

    cur_rm = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, cur_rm);
    morton_unshuffle(&plan, l, values + lev.beg, cur_rm);

    for (uint64_t i = 0; i < cur_n; ++i) {
      if (fabsf(cur_rm[i] - ref[i]) > 1e-5f) {
        uint64_t coords[MAX_NDIM];
        linear_to_coords(ndim, cur_shape, i, coords);
        printf("  FAIL level %d at (", l);
        for (int d = 0; d < ndim; ++d)
          printf("%s%llu", d ? "," : "", (unsigned long long)coords[d]);
        printf("): got %f, expected %f\n", cur_rm[i], ref[i]);
        goto Fail;
      }
    }
    printf("  level %d: ok\n", l);

    free(ref);
    ref = NULL;
    free(prev_rm);
    prev_rm = cur_rm;
    cur_rm = NULL;
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(values);
  lod_plan_free(&plan);
  free(prev_rm);
  free(ref);
  free(cur_rm);
  return ok;
}

int
main(void)
{
  int nfail = 0;
  nfail += !test_3d();
  nfail += !test_1d();

  // All dims downsampled (same as original tests)
  nfail +=
    !test_lod("test_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3);
  nfail +=
    !test_lod("test_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7);

  // Mixed: only some dims downsampled
  // 3D, only dims 0 and 2 downsampled (dim 1 is batch)
  nfail +=
    !test_lod("test_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5);
  // 3D, only dim 1 downsampled (dims 0 and 2 are batch)
  nfail +=
    !test_lod("test_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2);
  // 2D, only dim 0 downsampled (dim 1 is batch)
  nfail +=
    !test_lod("test_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1);
  // 2D, only dim 1 downsampled (dim 0 is batch)
  nfail +=
    !test_lod("test_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2);

  // No dims downsampled (trivial: nlev=1)
  nfail +=
    !test_lod("test_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0);
  // 1D downsampled
  nfail +=
    !test_lod("test_lod_1d", 1, (uint64_t[]){ 9 }, 0x1);

  // Larger mixed case
  nfail +=
    !test_lod("test_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);
  return nfail ? 1 : 0;
}
