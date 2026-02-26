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
  int ndim;
  int nlev;
  uint64_t shapes[MAX_LOD][MAX_NDIM];
  struct spans levels; // exclusive end of level k in values buf
  uint64_t* ends;      // contiguous child-group segment ends (absolute offsets)
};

// Segment slice for level l's child-group ends in the ends buffer.
// Derived from levels: segment l has counts[l+1] entries, and the segment
// layout is the level layout shifted down by one level.
static struct slice
lod_segment(const struct lod_plan* p, int level)
{
  struct slice next = spans_at(&p->levels, level + 1);
  uint64_t base = p->levels.ends[0]; // = counts[0]
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
// contiguous values buffer (absolute offset).
static void
lod_fill_ends(int ndim,
              const uint64_t* child_shape,
              const uint64_t* parent_shape,
              uint64_t n_parents,
              uint64_t val_base,
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

    ends[pos] = val_base + val;
  }
}

// Compute the shape pyramid, buffer layout, and child-group segment ends.
// Returns 0 on failure.
static int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              int max_levels)
{
  memset(p, 0, sizeof(*p));
  p->ndim = ndim;

  memcpy(p->shapes[0], shape, (size_t)ndim * sizeof(uint64_t));
  p->nlev = 1;
  while (p->nlev < max_levels && !is_all_ones(ndim, p->shapes[p->nlev - 1])) {
    for (int d = 0; d < ndim; ++d)
      p->shapes[p->nlev][d] = (p->shapes[p->nlev - 1][d] + 1) / 2;
    ++p->nlev;
  }

  // Element counts per level
  uint64_t counts[MAX_LOD];
  for (int k = 0; k < p->nlev; ++k) {
    uint64_t c = 1;
    for (int d = 0; d < ndim; ++d)
      c *= p->shapes[k][d];
    counts[k] = c;
  }

  // levels: partitions the values buffer by LOD level
  p->levels.n = (uint64_t)p->nlev;
  p->levels.ends = (uint64_t*)malloc(p->nlev * sizeof(uint64_t));
  if (!p->levels.ends)
    goto Fail;
  p->levels.ends[0] = counts[0];
  for (int k = 1; k < p->nlev; ++k)
    p->levels.ends[k] = p->levels.ends[k - 1] + counts[k];

  // Compute child-group segment ends for all levels.
  // Total ends = sum(counts[1..nlev-1]) = total_vals - counts[0].
  {
    uint64_t total_ends = p->levels.ends[p->nlev - 1] - p->levels.ends[0];
    if (total_ends > 0) {
      p->ends = (uint64_t*)malloc(total_ends * sizeof(uint64_t));
      if (!p->ends)
        goto Fail;
      for (int l = 0; l < p->nlev - 1; ++l) {
        struct slice seg = lod_segment(p, l);
        struct slice lev = spans_at(&p->levels, l);
        lod_fill_ends(ndim,
                      p->shapes[l],
                      p->shapes[l + 1],
                      slice_len(seg),
                      lev.beg,
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
  free(p->ends);
  memset(p, 0, sizeof(*p));
}

// Scatter array (row-major) into compacted Morton order for level 0.
static void
lod_scatter(const struct lod_plan* p, const float* src, float* dst)
{
  int ndim = p->ndim;
  const uint64_t* shape = p->shapes[0];
  uint64_t n = slice_len(spans_at(&p->levels, 0));

  uint64_t coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(ndim, shape, i, coords);
    uint64_t pos = morton_rank(ndim, shape, coords, 0);
    dst[pos] = src[i];
  }
}

// Segmented mean reduction over all LOD levels in one pass.
// The ends array's absolute offsets align at level boundaries, so a flat
// left-to-right scan naturally respects inter-level dependencies.
static void
lod_reduce(const struct lod_plan* p, float* values)
{
  uint64_t total_ends = p->levels.ends[p->nlev - 1] - p->levels.ends[0];
  uint64_t dst_base = p->levels.ends[0];

  for (uint64_t i = 0; i < total_ends; ++i) {
    uint64_t start = i > 0 ? p->ends[i - 1] : 0;
    uint64_t end = p->ends[i];
    uint64_t len = end - start;
    float sum = 0;
    for (uint64_t j = start; j < end; ++j)
      sum += values[j];
    values[dst_base + i] = sum / (float)len;
  }
}

// Compute all LOD levels into a contiguous values buffer.
// Plan must be initialized via lod_plan_init before calling.
// *out_values: contiguous buffer [level0][level1]...[levelN-1] in compacted
//              Morton order.
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

// Reference: brute-force downsample by averaging only valid children in each
// 2^d block. This matches lod_reduce's semantics: boundary parents have fewer
// than 2^d valid children and the mean is taken over the valid ones only.
// (This is equivalent to replicate-BC averaging because replicated values are
// copies of boundary elements, so the weighted and unweighted means coincide.)
static void
downsample_ref(int ndim,
               const uint64_t* cur_shape,
               const uint64_t* next_shape,
               const float* src,
               float* dst)
{
  uint64_t n_next = 1;
  for (int d = 0; d < ndim; ++d)
    n_next *= next_shape[d];

  uint64_t coords[MAX_NDIM];
  for (uint64_t j = 0; j < n_next; ++j) {
    linear_to_coords(ndim, next_shape, j, coords);

    float sum = 0;
    int count = 0;
    int n_children = 1 << ndim;
    for (int c = 0; c < n_children; ++c) {
      uint64_t lin = 0;
      uint64_t stride = 1;
      int valid = 1;
      for (int d = 0; d < ndim; ++d) {
        uint64_t child_coord = coords[d] * 2 + ((c >> d) & 1);
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

// Unshuffle: convert from compacted Morton order back to row-major.
static void
morton_unshuffle(int ndim,
                 const uint64_t* shape,
                 const float* morton_buf,
                 float* rowmajor)
{
  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  uint64_t coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(ndim, shape, i, coords);
    uint64_t pos = morton_rank(ndim, shape, coords, 0);
    rowmajor[i] = morton_buf[pos];
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
test_lod(const char* label, int ndim, const uint64_t* shape)
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

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, MAX_LOD));
  CHECK(Fail, lod_compute(&plan, src, &values));
  printf("  levels: %d\n", plan.nlev);
  CHECK(Fail, plan.nlev >= 2);

  // Verify level 0 scatter roundtrips
  prev_rm = (float*)malloc(n * sizeof(float));
  CHECK(Fail, prev_rm);
  morton_unshuffle(ndim, shape, values, prev_rm);
  for (uint64_t i = 0; i < n; ++i) {
    if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
      printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
             (unsigned long long)i,
             prev_rm[i],
             src[i]);
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
    downsample_ref(ndim, prev_shape, cur_shape, prev_rm, ref);

    cur_rm = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, cur_rm);
    morton_unshuffle(ndim, cur_shape, values + lev.beg, cur_rm);

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
  nfail += !test_lod("test_lod_2d", 2, (uint64_t[]){ 3, 5 });
  nfail += !test_lod("test_lod_3d", 3, (uint64_t[]){ 3, 2, 5 });
  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);
  return nfail ? 1 : 0;
}
