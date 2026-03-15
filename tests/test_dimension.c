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
  // defaults: tile_size = size
  CHECK(Error, dims[0].tile_size == 1000);
  CHECK(Error, dims[1].tile_size == 2);
  // defaults: identity storage_position
  CHECK(Error, dims[0].storage_position == 0);
  CHECK(Error, dims[1].storage_position == 1);
  CHECK(Error, dims[2].storage_position == 2);
  CHECK(Error, dims[3].storage_position == 3);
  // defaults: no downsample
  CHECK(Error, dims[0].downsample == 0);
  // defaults: tiles_per_shard = 0
  CHECK(Error, dims[0].tiles_per_shard == 0);

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
test_dims_budget_tile_size(void)
{
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 1000, 2, 2048, 2304 };
  dims_create(dims, "tcyx", sizes);

  // nelem=1<<19, ratios=[1,0,2,2], sum=5
  // bits_per_part = ceil(19/5) = 4, remainder = 19 - 20 = -1
  // dim 0 (ratio 1, first nonzero): 1<<(1*4-1) = 1<<3 = 8
  // dim 1 (ratio 0): 1
  // dim 2 (ratio 2): 1<<(2*4)   = 1<<8 = 256
  // dim 3 (ratio 2): 1<<(2*4)   = 1<<8 = 256
  uint8_t ratios[] = { 1, 0, 2, 2 };
  dims_budget_tile_size(dims, 4, 1ULL << 19, ratios);

  CHECK(Error, dims[0].tile_size == 8);
  CHECK(Error, dims[1].tile_size == 1);
  CHECK(Error, dims[2].tile_size == 256);
  CHECK(Error, dims[3].tile_size == 256);

  // Verify total: 128 * 1 * 64 * 64 = 2^19
  uint64_t total = 1;
  for (int i = 0; i < 4; ++i)
    total *= dims[i].tile_size;
  CHECK(Error, total == (1ULL << 19));

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dims_budget_tile_size_uniform(void)
{
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 256, 256, 256 };
  dims_create(dims, "xyz", sizes);

  // nelem=1<<12, ratios=[1,1,1], sum=3
  // bits_per_part = ceil(12/3) = 4, remainder = 0
  // each dim: 1<<4 = 16
  uint8_t ratios[] = { 1, 1, 1 };
  dims_budget_tile_size(dims, 3, 1ULL << 12, ratios);

  CHECK(Error, dims[0].tile_size == 16);
  CHECK(Error, dims[1].tile_size == 16);
  CHECK(Error, dims[2].tile_size == 16);

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
  dims_budget_tile_size(dims, 4, 1ULL << 19, ratios);
  // tile_sizes: 8, 1, 256, 256
  // tile_counts: ceil(1000/8)=125, ceil(2/1)=2, ceil(2048/256)=8,
  // ceil(2304/256)=9

  uint64_t shard_counts[] = { 1, 1, 4, 4 };
  dims_set_shard_counts(dims, 4, shard_counts);

  // tps[0] = ceil(125/1) = 125
  CHECK(Error, dims[0].tiles_per_shard == 125);
  // tps[1] = ceil(2/1) = 2
  CHECK(Error, dims[1].tiles_per_shard == 2);
  // tps[2] = ceil(8/4) = 2
  CHECK(Error, dims[2].tiles_per_shard == 2);
  // tps[3] = ceil(9/4) = 3
  CHECK(Error, dims[3].tiles_per_shard == 3);

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
  dims[0].tile_size = 10;
  dims[1].tile_size = 20;
  dims[0].tiles_per_shard = 99; // should not change

  uint64_t shard_counts[] = { 0, 2 };
  dims_set_shard_counts(dims, 2, shard_counts);

  CHECK(Error, dims[0].tiles_per_shard == 99); // unchanged
  CHECK(Error, dims[1].tiles_per_shard == 5);  // ceil(10/2)

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
  dims_budget_tile_size(dims, 4, 1ULL << 19, ratios);

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
  int nerr = 0;
  nerr += test_dims_create();
  nerr += test_dims_create_errors();
  nerr += test_dims_budget_tile_size();
  nerr += test_dims_budget_tile_size_uniform();
  nerr += test_dims_set_shard_counts();
  nerr += test_dims_set_shard_counts_skip_zero();
  nerr += test_dims_set_storage_order();
  nerr += test_dims_set_downsample_by_name();
  nerr += test_dims_print();
  return nerr;
}
