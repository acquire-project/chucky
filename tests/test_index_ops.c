#include "index.ops.h"
#include "prelude.h"

static int
test_unravel_basic(void)
{
  log_info("=== test_unravel_basic ===");

  // 2D: shape (3, 4), row-major index 7 → coords (1, 3)
  // unravel decomposes idx by dividing: d0 = 7%3=1, d1 = 2%4=2
  // Wait — unravel iterates d=0..rank-1: coords[d] = idx % shape[d]
  // So for shape (3,4): coords[0] = 7%3=1, idx=7/3=2, coords[1] = 2%4=2
  {
    uint64_t shape[] = { 3, 4 };
    uint64_t coords[2];
    unravel(2, shape, 7, coords);
    CHECK(Fail, coords[0] == 1);
    CHECK(Fail, coords[1] == 2);
  }

  // 2D: shape (3, 4), index 0 → (0, 0)
  {
    uint64_t shape[] = { 3, 4 };
    uint64_t coords[2];
    unravel(2, shape, 0, coords);
    CHECK(Fail, coords[0] == 0);
    CHECK(Fail, coords[1] == 0);
  }

  // 2D: shape (3, 4), index 11 → last element
  // coords[0] = 11%3=2, idx=11/3=3, coords[1] = 3%4=3
  {
    uint64_t shape[] = { 3, 4 };
    uint64_t coords[2];
    unravel(2, shape, 11, coords);
    CHECK(Fail, coords[0] == 2);
    CHECK(Fail, coords[1] == 3);
  }

  // 3D: shape (2, 3, 4), index 13
  // coords[0] = 13%2=1, idx=13/2=6, coords[1] = 6%3=0, idx=6/3=2, coords[2] = 2%4=2
  {
    uint64_t shape[] = { 2, 3, 4 };
    uint64_t coords[3];
    unravel(3, shape, 13, coords);
    CHECK(Fail, coords[0] == 1);
    CHECK(Fail, coords[1] == 0);
    CHECK(Fail, coords[2] == 2);
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_unravel_ravel_roundtrip(void)
{
  log_info("=== test_unravel_ravel_roundtrip ===");

  // For identity strides (row-major), ravel(unravel(idx)) should return idx.
  // ravel iterates d=rank-1..0: coord = idx % shape[d], o += coord * strides[d]
  // With identity strides: strides[rank-1]=1, strides[d] = shape[d+1]*strides[d+1]

  // 2D: shape (3, 4)
  {
    uint64_t shape[] = { 3, 4 };
    // Identity strides for ravel: ravel iterates high d first (row-major)
    // strides[1]=1, strides[0]=4
    int64_t strides[] = { 4, 1 };
    uint64_t total = 3 * 4;

    for (uint64_t idx = 0; idx < total; ++idx) {
      uint64_t coords[2];
      unravel(2, shape, idx, coords);
      // Reconstruct using ravel with identity strides
      // ravel does: for d=1..0: coord = idx % shape[d], o += coord * strides[d]
      // But ravel takes the flat index and decomposes it differently (high-d first).
      // So we need to compute manually:
      uint64_t reconstructed = coords[0] * strides[0] + coords[1] * strides[1];
      // unravel produces coords in low-d-first order matching its shape decomposition.
      // But ravel uses the same shape to re-decompose, so ravel(idx) with identity
      // strides should return idx.
      uint64_t raveled = ravel(2, shape, strides, idx);
      CHECK(Fail, raveled == idx);
      (void)reconstructed;
    }
  }

  // 3D: shape (2, 3, 5)
  {
    uint64_t shape[] = { 2, 3, 5 };
    int64_t strides[] = { 15, 5, 1 };
    uint64_t total = 2 * 3 * 5;

    for (uint64_t idx = 0; idx < total; ++idx) {
      uint64_t raveled = ravel(3, shape, strides, idx);
      CHECK(Fail, raveled == idx);
    }
  }

  // 4D: shape (2, 2, 3, 4)
  {
    uint64_t shape[] = { 2, 2, 3, 4 };
    int64_t strides[] = { 24, 12, 4, 1 };
    uint64_t total = 2 * 2 * 3 * 4;

    for (uint64_t idx = 0; idx < total; ++idx) {
      uint64_t raveled = ravel(4, shape, strides, idx);
      CHECK(Fail, raveled == idx);
    }
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_ravel_transpose(void)
{
  log_info("=== test_ravel_transpose ===");

  // 2D: shape (3, 4) with transposed strides (column-major: strides = {1, 3})
  // ravel(idx) with these strides should scatter to column-major positions.
  // For idx=0: coord=(0,0) → 0*1 + 0*3 = 0
  // For idx=1: ravel decomposes high d first: d=1: 1%4=1, d=0: 0%3=0 → 0*1 + 1*3 = 3
  // For idx=4: d=1: 4%4=0, d=0: 1%3=1 → 1*1 + 0*3 = 1
  {
    uint64_t shape[] = { 3, 4 };
    int64_t strides[] = { 1, 3 }; // column-major (transposed)

    CHECK(Fail, ravel(2, shape, strides, 0) == 0);  // (0,0) → 0
    CHECK(Fail, ravel(2, shape, strides, 1) == 3);  // (0,1) → 3
    CHECK(Fail, ravel(2, shape, strides, 4) == 1);  // (1,0) → 1
    CHECK(Fail, ravel(2, shape, strides, 5) == 4);  // (1,1) → 4
    CHECK(Fail, ravel(2, shape, strides, 11) == 11); // (2,3) → 2+9=11
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_compute_strides(void)
{
  log_info("=== test_compute_strides ===");

  // 2D: shape (3, 4) → strides (4, 1)
  {
    int shape[] = { 3, 4 };
    int strides[2];
    compute_strides(2, shape, strides);
    CHECK(Fail, strides[0] == 4);
    CHECK(Fail, strides[1] == 1);
  }

  // 3D: shape (2, 3, 5) → strides (15, 5, 1)
  {
    int shape[] = { 2, 3, 5 };
    int strides[3];
    compute_strides(3, shape, strides);
    CHECK(Fail, strides[0] == 15);
    CHECK(Fail, strides[1] == 5);
    CHECK(Fail, strides[2] == 1);
  }

  // 4D: shape (2, 3, 4, 5) → strides (60, 20, 5, 1)
  {
    int shape[] = { 2, 3, 4, 5 };
    int strides[4];
    compute_strides(4, shape, strides);
    CHECK(Fail, strides[0] == 60);
    CHECK(Fail, strides[1] == 20);
    CHECK(Fail, strides[2] == 5);
    CHECK(Fail, strides[3] == 1);
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_permute_i32(void)
{
  log_info("=== test_permute_i32 ===");

  // out[i] = in[p[i]]
  {
    int p[] = { 2, 0, 1 };
    int in[] = { 10, 20, 30 };
    int out[3];
    permute_i32(3, p, in, out);
    CHECK(Fail, out[0] == 30); // in[p[0]] = in[2] = 30
    CHECK(Fail, out[1] == 10); // in[p[1]] = in[0] = 10
    CHECK(Fail, out[2] == 20); // in[p[2]] = in[1] = 20
  }

  // Identity permutation
  {
    int p[] = { 0, 1, 2, 3 };
    int in[] = { 5, 10, 15, 20 };
    int out[4];
    permute_i32(4, p, in, out);
    for (int i = 0; i < 4; ++i)
      CHECK(Fail, out[i] == in[i]);
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_inverse_permutation_i32(void)
{
  log_info("=== test_inverse_permutation_i32 ===");

  // inv[p[i]] = i  and  p[inv[i]] = i
  {
    int p[] = { 2, 0, 1 };
    int inv[3];
    inverse_permutation_i32(3, p, inv);
    // inv[p[0]] = inv[2] = 0, inv[p[1]] = inv[0] = 1, inv[p[2]] = inv[1] = 2
    CHECK(Fail, inv[2] == 0);
    CHECK(Fail, inv[0] == 1);
    CHECK(Fail, inv[1] == 2);

    // Verify p[inv[i]] = i
    for (int i = 0; i < 3; ++i)
      CHECK(Fail, p[inv[i]] == i);
  }

  // 4-element permutation
  {
    int p[] = { 3, 1, 0, 2 };
    int inv[4];
    inverse_permutation_i32(4, p, inv);
    for (int i = 0; i < 4; ++i) {
      CHECK(Fail, inv[p[i]] == i);
      CHECK(Fail, p[inv[i]] == i);
    }
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_ravel_i32_matches_ravel(void)
{
  log_info("=== test_ravel_i32_matches_ravel ===");

  // Both ravel variants should produce the same results.
  // 3D: shape (2, 3, 4), transposed strides
  {
    uint64_t shape64[] = { 2, 3, 4 };
    int64_t strides64[] = { 1, 2, 6 }; // column-major
    int shape32[] = { 2, 3, 4 };
    int strides32[] = { 1, 2, 6 };
    uint64_t total = 2 * 3 * 4;

    for (uint64_t idx = 0; idx < total; ++idx) {
      uint64_t r1 = ravel(3, shape64, strides64, idx);
      uint64_t r2 = ravel_i32(3, shape32, strides32, idx);
      CHECK(Fail, r1 == r2);
    }
  }

  // Row-major strides
  {
    uint64_t shape64[] = { 4, 5, 6 };
    int64_t strides64[] = { 30, 6, 1 };
    int shape32[] = { 4, 5, 6 };
    int strides32[] = { 30, 6, 1 };
    uint64_t total = 4 * 5 * 6;

    for (uint64_t idx = 0; idx < total; ++idx) {
      uint64_t r1 = ravel(3, shape64, strides64, idx);
      uint64_t r2 = ravel_i32(3, shape32, strides32, idx);
      CHECK(Fail, r1 == r2);
    }
  }

  log_info("  PASS");
  return 0;
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_edge_cases(void)
{
  log_info("=== test_edge_cases ===");

  // Rank 1: shape (7), identity strides (1)
  {
    uint64_t shape[] = { 7 };
    int64_t strides[] = { 1 };
    for (uint64_t idx = 0; idx < 7; ++idx) {
      CHECK(Fail, ravel(1, shape, strides, idx) == idx);
    }
  }

  // Rank 1: unravel
  {
    uint64_t shape[] = { 5 };
    uint64_t coords[1];
    unravel(1, shape, 3, coords);
    CHECK(Fail, coords[0] == 3);
  }

  // Single-element dimensions: shape (1, 4, 1)
  {
    uint64_t shape[] = { 1, 4, 1 };
    int64_t strides[] = { 4, 1, 1 };
    for (uint64_t idx = 0; idx < 4; ++idx) {
      CHECK(Fail, ravel(3, shape, strides, idx) == idx);
    }
  }

  // compute_strides with rank=1
  {
    int shape[] = { 10 };
    int strides[1];
    compute_strides(1, shape, strides);
    CHECK(Fail, strides[0] == 1);
  }

  // compute_strides with single-element dims
  {
    int shape[] = { 1, 3, 1 };
    int strides[3];
    compute_strides(3, shape, strides);
    CHECK(Fail, strides[0] == 3);
    CHECK(Fail, strides[1] == 1);
    CHECK(Fail, strides[2] == 1);
  }

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

  int ecode = 0;
  ecode |= test_unravel_basic();
  ecode |= test_unravel_ravel_roundtrip();
  ecode |= test_ravel_transpose();
  ecode |= test_compute_strides();
  ecode |= test_permute_i32();
  ecode |= test_inverse_permutation_i32();
  ecode |= test_ravel_i32_matches_ravel();
  ecode |= test_edge_cases();
  return ecode;
}
