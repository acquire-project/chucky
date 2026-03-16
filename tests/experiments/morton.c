#include "morton.util.c"

static int
ceil_log2_local(uint64_t v)
{
  int p = 0;
  while ((1ull << p) < v)
    ++p;
  return p;
}

static uint64_t
max_shape_local(int ndim, const uint64_t* shape)
{
  uint64_t m = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > m)
      m = shape[d];
  return m;
}

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
  uint64_t lod_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    // C-order decomposition (dim ndim-1 fastest)
    {
      uint64_t rest = i;
      for (int d = p->ndim - 1; d >= 0; --d) {
        full_coords[d] = rest % full_shape[d];
        rest /= full_shape[d];
      }
    }
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_lod(p, full_coords, lod_coords);
    uint64_t pos =
      morton_rank(p->lod_ndim, p->lod_shapes[level], lod_coords, 0);
    rowmajor[i] = morton_buf[b * p->lod_counts[level] + pos];
  }
}

static void
downsample_ref(int ndim,
               uint32_t ds_mask,
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
    // C-order decomposition (dim ndim-1 fastest)
    {
      uint64_t rest = j;
      for (int d = ndim - 1; d >= 0; --d) {
        coords[d] = rest % next_shape[d];
        rest /= next_shape[d];
      }
    }

    float sum = 0;
    int count = 0;
    for (int c = 0; c < n_children; ++c) {
      uint64_t lin = 0;
      uint64_t stride = 1;
      int valid = 1;
      int ds_bit = 0;

      // Compute lin in C-order (dim ndim-1 fastest)
      for (int d = ndim - 1; d >= 0; --d) {
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

  int p = ceil_log2_local(max_shape_local(ndim, shape));
  uint64_t box = 1ull << (ndim * p);
  for (uint64_t k = 0; k <= box; ++k) {
    uint64_t coords[MAX_NDIM];
    morton_decode(ndim, k, coords);
    uint64_t r = morton_rank(ndim, shape, coords, 0);
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
test_lod(const char* label, int ndim, const uint64_t* shape, uint32_t lod_mask)
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

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD) == 0);
  printf("  lod_mask=0x%x  lod_ndim=%d  batch_ndim=%d  batch_count=%llu\n",
         lod_mask,
         plan.lod_ndim,
         plan.batch_ndim,
         (unsigned long long)plan.batch_count);

  CHECK(Fail, lod_compute(&plan, src, &values, lod_reduce_mean));
  printf("  levels: %d\n", plan.nlod);

  if (plan.nlod < 2) {
    prev_rm = (float*)malloc(n * sizeof(float));
    CHECK(Fail, prev_rm);
    morton_unshuffle(&plan, 0, values, prev_rm);
    for (uint64_t i = 0; i < n; ++i) {
      if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
        printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
               (unsigned long long)i,
               prev_rm[i],
               src[i]);
        goto Fail;
      }
    }
    printf("  level 0 scatter: ok (no downsample levels)\n");
    printf("  PASS\n");
    ok = 1;
    goto Fail;
  }

  CHECK(Fail, plan.nlod >= 2);

  prev_rm = (float*)malloc(n * sizeof(float));
  CHECK(Fail, prev_rm);
  morton_unshuffle(&plan, 0, values, prev_rm);
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

  for (int l = 1; l < plan.nlod; ++l) {
    const uint64_t* prev_shape = plan.shapes[l - 1];
    const uint64_t* cur_shape = plan.shapes[l];
    struct lod_span lev = lod_spans_at(&plan.levels, l);
    uint64_t cur_n = lod_span_len(lev);

    ref = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, ref);
    downsample_ref(ndim, lod_mask, prev_shape, cur_shape, prev_rm, ref);

    cur_rm = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, cur_rm);
    morton_unshuffle(&plan, l, values + lev.beg, cur_rm);

    for (uint64_t i = 0; i < cur_n; ++i) {
      if (fabsf(cur_rm[i] - ref[i]) > 1e-5f) {
        uint64_t coords[MAX_NDIM];
        unravel(ndim, cur_shape, i, coords);
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

  nfail += !test_lod("test_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3);
  nfail += !test_lod("test_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7);

  nfail += !test_lod("test_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5);
  nfail += !test_lod("test_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2);
  nfail += !test_lod("test_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1);
  nfail += !test_lod("test_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2);

  nfail += !test_lod("test_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0);
  nfail += !test_lod("test_lod_1d", 1, (uint64_t[]){ 9 }, 0x1);

  nfail += !test_lod("test_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);
  return nfail ? 1 : 0;
}
