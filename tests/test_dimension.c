#include "dimension.h"
#include "prelude.h"
#include "stream.h"

#include <stdio.h>
#include <string.h>

#define REPORT_TEST(ok)                                                        \
  do {                                                                         \
    if (ok)                                                                    \
      log_info("  PASS: %s", __func__);                                        \
    else                                                                       \
      log_error("  FAIL: %s", __func__);                                       \
  } while (0)

static int
test_dims_create(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  uint8_t rank = dims_create(dims, "tcyx", sizes);

  CHECK(Error, rank == 4);
  CHECK(Error, dims[0].name[0] == 't');
  CHECK(Error, dims[1].name[0] == 'c');
  CHECK(Error, dims[2].name[0] == 'y');
  CHECK(Error, dims[3].name[0] == 'x');
  CHECK(Error, dims[0].size == 1000);
  CHECK(Error, dims[1].size == 2);
  CHECK(Error, dims[2].size == 2048);
  CHECK(Error, dims[3].size == 2304);
  // defaults: chunk_size = size
  CHECK(Error, dims[0].chunk_size == 1000);
  CHECK(Error, dims[1].chunk_size == 2);
  // defaults: identity storage_position
  CHECK(Error, dims[0].storage_position == 0);
  CHECK(Error, dims[1].storage_position == 1);
  CHECK(Error, dims[2].storage_position == 2);
  CHECK(Error, dims[3].storage_position == 3);
  // defaults: no downsample
  CHECK(Error, dims[0].downsample == 0);
  // defaults: chunks_per_shard = 0
  CHECK(Error, dims[0].chunks_per_shard == 0);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_create_errors(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 100 };

  CHECK(Error, dims_create(NULL, "x", sizes) == 0);
  CHECK(Error, dims_create(dims, NULL, sizes) == 0);
  CHECK(Error, dims_create(dims, "", sizes) == 0);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_budget_chunk_size(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  // nelem=1<<19, ratios=[1,0,2,2], sum=5
  // Greedy bit allocation (higher-index tiebreak):
  // dim 0 (ratio 1): 4 bits -> 16
  // dim 1 (ratio 0): 0 bits -> 1
  // dim 2 (ratio 2): 7 bits -> 128
  // dim 3 (ratio 2): 8 bits -> 256
  uint8_t ratios[] = { 1, 0, 2, 2 };
  dims_budget_chunk_size(dims, 4, 1ULL << 19, ratios);

  CHECK(Error, dims[0].chunk_size == 16);
  CHECK(Error, dims[1].chunk_size == 1);
  CHECK(Error, dims[2].chunk_size == 128);
  CHECK(Error, dims[3].chunk_size == 256);

  // Verify total: 16 * 1 * 128 * 256 = 2^19
  uint64_t total = 1;
  for (int i = 0; i < 4; ++i)
    total *= dims[i].chunk_size;
  CHECK(Error, total == (1ULL << 19));

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_budget_chunk_size_uniform(void)
{
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 256, 256, 256 };
  dims_create(dims, "xyz", sizes);

  // nelem=1<<12, ratios=[1,1,1], sum=3
  // bits_per_part = ceil(12/3) = 4, remainder = 0
  // each dim: 1<<4 = 16
  uint8_t ratios[] = { 1, 1, 1 };
  dims_budget_chunk_size(dims, 3, 1ULL << 12, ratios);

  CHECK(Error, dims[0].chunk_size == 16);
  CHECK(Error, dims[1].chunk_size == 16);
  CHECK(Error, dims[2].chunk_size == 16);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_set_shard_counts(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  uint8_t ratios[] = { 1, 0, 2, 2 };
  dims_budget_chunk_size(dims, 4, 1ULL << 19, ratios);
  // chunk_sizes: 16, 1, 128, 256
  // chunk_counts: ceil(1000/16)=63, ceil(2/1)=2, ceil(2048/128)=16,
  // ceil(2304/256)=9

  uint64_t shard_counts[] = { 1, 1, 4, 4 };
  dims_set_shard_counts(dims, 4, shard_counts);

  // tps[0] = ceil(63/1) = 63
  CHECK(Error, dims[0].chunks_per_shard == 63);
  // tps[1] = ceil(2/1) = 2
  CHECK(Error, dims[1].chunks_per_shard == 2);
  // tps[2] = ceil(16/4) = 4
  CHECK(Error, dims[2].chunks_per_shard == 4);
  // tps[3] = ceil(9/4) = 3
  CHECK(Error, dims[3].chunks_per_shard == 3);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_set_shard_counts_skip_zero(void)
{
  int ok = 0;
  struct dimension dims[2];
  uint64_t sizes[] = { 100, 200 };
  dims_create(dims, "xy", sizes);
  dims[0].chunk_size = 10;
  dims[1].chunk_size = 20;
  dims[0].chunks_per_shard = 99; // should not change

  uint64_t shard_counts[] = { 0, 2 };
  dims_set_shard_counts(dims, 2, shard_counts);

  CHECK(Error, dims[0].chunks_per_shard == 99); // unchanged
  CHECK(Error, dims[1].chunks_per_shard == 5);  // ceil(10/2)

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_budget_chunk_bytes(void)
{
  int ok = 0;
  struct dimension dims_a[4], dims_b[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims_a, "tcyx", sizes);
  dims_create(dims_b, "tcyx", sizes);

  uint8_t ratios[] = { 1, 0, 2, 2 };
  // 1MB / 2 bytes_per_element = 2^19 elements
  dims_budget_chunk_bytes(dims_a, 4, 1 << 20, 2, ratios);
  dims_budget_chunk_size(dims_b, 4, 1ULL << 19, ratios);

  for (int i = 0; i < 4; ++i)
    CHECK(Error, dims_a[i].chunk_size == dims_b[i].chunk_size);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

struct advise_test_ctx
{
  struct dimension* dims;
  uint8_t rank;
  size_t bpe;
};

static int
advise_estimate(void* ctx, size_t* out)
{
  struct advise_test_ctx* c = (struct advise_test_ctx*)ctx;
  size_t total = 1;
  for (uint8_t i = 0; i < c->rank; ++i)
    total *= c->dims[i].chunk_size;
  // Simulate: memory = 10 * chunk_volume * bpe
  *out = 10 * total * c->bpe;
  return 0;
}

static int
test_dims_advise(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  uint8_t ratios[] = { 1, 0, 2, 2 };
  size_t bpe = 2;
  struct advise_test_ctx actx = { .dims = dims, .rank = 4, .bpe = bpe };

  // Budget: 10 * 2^19 * 2 = 10 MiB. Starting from 1MB chunks (2^19 elems).
  // estimate(1MB) = 10 * 2^19 * 2 = 10 MiB <= 10 MiB => fits on first try
  size_t budget = 10 * (1ULL << 19) * bpe;
  CHECK(Error, dims_advise(dims, 4, 1 << 20, bpe, ratios, budget,
                           advise_estimate, &actx) == 0);
  // Should have used the full 1MB target
  uint64_t vol = 1;
  for (int i = 0; i < 4; ++i)
    vol *= dims[i].chunk_size;
  CHECK(Error, vol == (1ULL << 19));

  // Now use a budget that's too small for 1MB but fits 512KB
  // estimate(1MB) = 10 * 2^19 * 2 = 10 MiB > 5 MiB
  // estimate(512KB) = 10 * 2^18 * 2 = 5 MiB <= 5 MiB => fits
  size_t smaller_budget = 10 * (1ULL << 18) * bpe;
  CHECK(Error, dims_advise(dims, 4, 1 << 20, bpe, ratios, smaller_budget,
                           advise_estimate, &actx) == 0);
  vol = 1;
  for (int i = 0; i < 4; ++i)
    vol *= dims[i].chunk_size;
  CHECK(Error, vol == (1ULL << 18));

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_set_storage_order(void)
{
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 10, 20, 30 };
  dims_create(dims, "xyz", sizes);

  // "xzy" means storage order x=0, z=1, y=2
  CHECK(Error, dims_set_storage_order(dims, 3, "xzy") == 0);
  CHECK(Error, dims[0].storage_position == 0); // x -> 0
  CHECK(Error, dims[1].storage_position == 2); // y -> 2
  CHECK(Error, dims[2].storage_position == 1); // z -> 1

  // NULL resets to identity
  CHECK(Error, dims_set_storage_order(dims, 3, NULL) == 0);
  CHECK(Error, dims[0].storage_position == 0);
  CHECK(Error, dims[1].storage_position == 1);
  CHECK(Error, dims[2].storage_position == 2);

  // Error: wrong length
  CHECK(Error, dims_set_storage_order(dims, 3, "xy") != 0);

  // Error: append dim not first
  CHECK(Error, dims_set_storage_order(dims, 3, "yxz") != 0);

  // Error: unknown dim name
  CHECK(Error, dims_set_storage_order(dims, 3, "xqz") != 0);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_set_downsample_by_name(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  dims_set_downsample_by_name(dims, 4, "yx");

  CHECK(Error, dims[0].downsample == 0); // t
  CHECK(Error, dims[1].downsample == 0); // c
  CHECK(Error, dims[2].downsample == 1); // y
  CHECK(Error, dims[3].downsample == 1); // x

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_print(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  uint8_t ratios[] = { 1, 0, 2, 2 };
  dims_budget_chunk_size(dims, 4, 1ULL << 19, ratios);

  uint64_t shard_counts[] = { 1, 1, 4, 4 };
  dims_set_shard_counts(dims, 4, shard_counts);

  dims_set_downsample_by_name(dims, 4, "yx");

  // Just verify it doesn't crash.
  dims_print(dims, 4);

  ok = 1;
  REPORT_TEST(ok);
  return !ok;
}

int
main(void)
{
  int rc = 0;
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "dims_create", test_dims_create },
    { "dims_create_errors", test_dims_create_errors },
    { "dims_budget_chunk_size", test_dims_budget_chunk_size },
    { "dims_budget_chunk_size_uniform", test_dims_budget_chunk_size_uniform },
    { "dims_budget_chunk_bytes", test_dims_budget_chunk_bytes },
    { "dims_advise", test_dims_advise },
    { "dims_set_shard_counts", test_dims_set_shard_counts },
    { "dims_set_shard_counts_skip_zero", test_dims_set_shard_counts_skip_zero },
    { "dims_set_storage_order", test_dims_set_storage_order },
    { "dims_set_downsample_by_name", test_dims_set_downsample_by_name },
    { "dims_print", test_dims_print },
  };
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    int r = tests[i].fn();
    if (r) {
      log_error("  FAIL: %s", tests[i].name);
      rc = 1;
    } else {
      log_info("  PASS: %s", tests[i].name);
    }
  }
  return rc;
}
