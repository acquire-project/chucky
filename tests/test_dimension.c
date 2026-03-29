#include "dimension.h"
#include "stream/dim_info.h"
#include "util/prelude.h"

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

// --- dim_info tests ---

static int
test_dim_info_single_append(void)
{
  // Standard case: n_append=1 (dim 0 has chunk_size > 1)
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 0, 64, 64 };
  dims_create(dims, "tyx", sizes);
  uint64_t cs[] = { 1, 32, 32 };
  dims_set_chunk_sizes(dims, 3, cs);
  dims[0].chunks_per_shard = 4;

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 3) == 0);
  CHECK(Error, dim_info_n_append(&info) == 1);
  CHECK(Error, dim_info_rank(&info) == 3);
  CHECK(Error, info.append.beg == &dims[0]);
  CHECK(Error, info.append.end == &dims[1]);
  CHECK(Error, info.inner.beg == &dims[1]);
  CHECK(Error, info.inner.end == &dims[3]);
  CHECK(Error, info.append_downsample == 0);
  CHECK(Error, info.lod_mask == 0);
  CHECK(Error, info.bounded_append_chunks == 1);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_two_append(void)
{
  // n_append=2: dims "tzyx", chunk (1,1,64,64)
  int ok = 0;
  struct dimension dims[4];
  uint64_t sizes[] = { 0, 10, 128, 128 };
  dims_create(dims, "tzyx", sizes);
  uint64_t cs[] = { 1, 1, 64, 64 };
  dims_set_chunk_sizes(dims, 4, cs);
  dims[0].chunks_per_shard = 4;
  dims[1].chunks_per_shard = 10;
  dims_set_downsample_by_name(dims, 4, "yx");

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 4) == 0);
  CHECK(Error, dim_info_n_append(&info) == 2);
  CHECK(Error, dim_info_rank(&info) == 4);

  // append = dims[0..2), inner = dims[2..4)
  CHECK(Error, dim_slice_len(info.append) == 2);
  CHECK(Error, dim_slice_len(info.inner) == 2);
  CHECK(Error, info.append.beg == &dims[0]);
  CHECK(Error, info.inner.beg == &dims[2]);

  // No downsample on append dims -> append_downsample=0
  CHECK(Error, info.append_downsample == 0);

  // LOD mask: dims 2,3 have downsample -> bits 2,3
  CHECK(Error, info.lod_mask == ((1u << 2) | (1u << 3)));

  // bounded_append_chunks = chunk_count for dim 1 = ceil(10/1) = 10
  CHECK(Error, info.bounded_append_chunks == 10);

  // dim_index works
  CHECK(Error, dim_index(&info, &dims[0]) == 0);
  CHECK(Error, dim_index(&info, &dims[2]) == 2);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_two_append_with_downsample(void)
{
  // n_append=2, rightmost append dim (z) downsampled
  // chunk (1,1,1,64,64), downsample on dims 1,3,4
  // max_n = 3 (dims 0,1,2 have chunk_size=1)
  // first downsample in prefix = dim 1 -> n_append = 2
  int ok = 0;
  struct dimension dims[5];
  uint64_t sizes[] = { 0, 10, 20, 128, 128 };
  dims_create(dims, "tzcyx", sizes);
  uint64_t cs[] = { 1, 1, 1, 64, 64 };
  dims_set_chunk_sizes(dims, 5, cs);
  dims[0].chunks_per_shard = 4;
  dims_set_downsample_by_name(dims, 5, "zyx");

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 5) == 0);
  CHECK(Error, dim_info_n_append(&info) == 2);

  // dim 1 (z) is rightmost append dim and has downsample
  CHECK(Error, info.append_downsample == 1);

  // LOD mask: dims 3,4 (y,x) have downsample, dim 2 (c) does not
  // dim 1 (z) is append, not in LOD mask
  CHECK(Error, info.lod_mask == ((1u << 3) | (1u << 4)));

  // bounded_append_chunks = chunk_count for dim 1 = ceil(10/1) = 10
  CHECK(Error, info.bounded_append_chunks == 10);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_three_append(void)
{
  // n_append=3: 5D "tzcyx", chunk (1,1,1,64,64)
  int ok = 0;
  struct dimension dims[5];
  uint64_t sizes[] = { 0, 8, 3, 128, 128 };
  dims_create(dims, "tzcyx", sizes);
  uint64_t cs[] = { 1, 1, 1, 64, 64 };
  dims_set_chunk_sizes(dims, 5, cs);
  dims[0].chunks_per_shard = 4;
  dims[1].chunks_per_shard = 8;
  dims[2].chunks_per_shard = 3;
  dims_set_downsample_by_name(dims, 5, "yx");

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 5) == 0);
  CHECK(Error, dim_info_n_append(&info) == 3);
  CHECK(Error, dim_info_rank(&info) == 5);

  // append = dims[0..3), inner = dims[3..5)
  CHECK(Error, dim_slice_len(info.append) == 3);
  CHECK(Error, dim_slice_len(info.inner) == 2);
  CHECK(Error, info.append.beg == &dims[0]);
  CHECK(Error, info.inner.beg == &dims[3]);

  // No downsample on any append dim
  CHECK(Error, info.append_downsample == 0);

  // LOD mask: dims 3,4 (y,x) have downsample
  CHECK(Error, info.lod_mask == ((1u << 3) | (1u << 4)));

  // bounded_append_chunks = chunk_count[1] * chunk_count[2]
  //   = ceil(8/1) * ceil(3/1) = 8 * 3 = 24
  CHECK(Error, info.bounded_append_chunks == 24);

  // dim_index works for all dims
  CHECK(Error, dim_index(&info, &dims[0]) == 0);
  CHECK(Error, dim_index(&info, &dims[2]) == 2);
  CHECK(Error, dim_index(&info, &dims[4]) == 4);

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_rejects_unbounded_non_dim0(void)
{
  int ok = 0;
  struct dimension dims[2];
  uint64_t sizes[] = { 100, 0 }; // dim 1 unbounded — invalid
  dims_create(dims, "xy", sizes);
  uint64_t cs[] = { 1, 1 };
  dims_set_chunk_sizes(dims, 2, cs);

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 2) != 0); // should fail

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_final_append_sizes(void)
{
  // Bug scenario: 12 frames with chunk_size=5 should report 12, not 15.
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 0, 64, 64 };
  dims_create(dims, "tyx", sizes);
  uint64_t cs[] = { 5, 32, 32 };
  dims_set_chunk_sizes(dims, 3, cs);
  dims[0].chunks_per_shard = 4;

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 3) == 0);

  // 12 frames of 64x64
  uint64_t cursor = 12 * 64 * 64;
  uint64_t append_sizes[1];

  // Level 0: exact dim0 = 12 (not ceildiv(12,5)*5 = 15)
  dim_info_final_append_sizes(&info, cursor, 0, append_sizes);
  CHECK(Error, append_sizes[0] == 12);

  // Without append_downsample, all levels report the same dim0
  dim_info_final_append_sizes(&info, cursor, 1, append_sizes);
  CHECK(Error, append_sizes[0] == 12);

  // n_append=2: "tzyx", chunk (1,1,64,64), z bounded at size=10
  {
    struct dimension dims2[4];
    uint64_t sizes2[] = { 0, 10, 128, 128 };
    dims_create(dims2, "tzyx", sizes2);
    uint64_t cs2[] = { 1, 1, 64, 64 };
    dims_set_chunk_sizes(dims2, 4, cs2);
    dims2[0].chunks_per_shard = 4;
    dims2[1].chunks_per_shard = 10;
    dims_set_downsample_by_name(dims2, 4, "yx");

    struct dim_info info2;
    CHECK(Error, dim_info_init(&info2, dims2, 4) == 0);
    CHECK(Error, dim_info_n_append(&info2) == 2);

    // 7 frames of 10×128×128
    uint64_t cursor2 = 7 * 10 * 128 * 128;
    uint64_t as2[2];
    dim_info_final_append_sizes(&info2, cursor2, 0, as2);
    CHECK(Error, as2[0] == 7);  // dim 0: exact from cursor
    CHECK(Error, as2[1] == 10); // dim 1: bounded, declared size
  }

  ok = 1;
Error:
  REPORT_TEST(ok);
  return !ok;
}

static int
test_dim_info_final_append_sizes_lod(void)
{
  // With append_downsample, LOD levels halve dim 0.
  int ok = 0;
  struct dimension dims[3];
  uint64_t sizes[] = { 0, 64, 64 };
  dims_create(dims, "tyx", sizes);
  uint64_t cs[] = { 1, 32, 32 };
  dims_set_chunk_sizes(dims, 3, cs);
  dims[0].chunks_per_shard = 4;
  dims_set_downsample_by_name(dims, 3, "tyx");

  struct dim_info info;
  CHECK(Error, dim_info_init(&info, dims, 3) == 0);
  CHECK(Error, info.append_downsample == 1);

  uint64_t cursor = 12 * 64 * 64;
  uint64_t append_sizes[1];

  // Level 0: exact
  dim_info_final_append_sizes(&info, cursor, 0, append_sizes);
  CHECK(Error, append_sizes[0] == 12);

  // Level 1: ceildiv(12, 2) = 6
  dim_info_final_append_sizes(&info, cursor, 1, append_sizes);
  CHECK(Error, append_sizes[0] == 6);

  // Level 2: ceildiv(12, 4) = 3
  dim_info_final_append_sizes(&info, cursor, 2, append_sizes);
  CHECK(Error, append_sizes[0] == 3);

  // Level 3: ceildiv(12, 8) = 2
  dim_info_final_append_sizes(&info, cursor, 3, append_sizes);
  CHECK(Error, append_sizes[0] == 2);

  // Odd count: 11 frames
  cursor = 11 * 64 * 64;
  dim_info_final_append_sizes(&info, cursor, 1, append_sizes);
  CHECK(Error, append_sizes[0] == 6); // ceildiv(11, 2)

  ok = 1;
Error:
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

    { "dims_set_shard_counts", test_dims_set_shard_counts },
    { "dims_set_shard_counts_skip_zero", test_dims_set_shard_counts_skip_zero },
    { "dims_set_storage_order", test_dims_set_storage_order },
    { "dims_set_downsample_by_name", test_dims_set_downsample_by_name },
    { "dims_print", test_dims_print },
    { "dim_info_single_append", test_dim_info_single_append },
    { "dim_info_two_append", test_dim_info_two_append },
    { "dim_info_two_append_with_downsample",
      test_dim_info_two_append_with_downsample },
    { "dim_info_three_append", test_dim_info_three_append },
    { "dim_info_rejects_unbounded_non_dim0",
      test_dim_info_rejects_unbounded_non_dim0 },
    { "dim_info_final_append_sizes", test_dim_info_final_append_sizes },
    { "dim_info_final_append_sizes_lod", test_dim_info_final_append_sizes_lod },
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
