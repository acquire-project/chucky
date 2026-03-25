#include "cpu/aggregate.h"
#include "util/index.ops.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// Build a simple aggregate layout for a 3D case:
//   chunk_count = {2, 3} (dims 1..rank-1, dim 0 is the epoch dim)
//   chunks_per_shard = {2, 3} (one shard covers everything)
//   chunks_per_epoch M = 6
// So C = 6 and the permutation is the identity for this case.
static int
test_simple(void)
{
  log_info("=== test_aggregate_cpu_simple ===");

  struct aggregate_layout layout;
  memset(&layout, 0, sizeof(layout));

  uint64_t chunk_count[] = { 0, 2, 3 };
  uint64_t chunks_per_shard[] = { 0, 2, 3 };
  uint8_t rank = 3; // rank includes dim 0
  uint64_t M = 6;
  size_t max_comp = 128;

  size_t comp_sizes[6];
  char* compressed = NULL;

  CHECK(Fail,
        aggregate_layout_compute(
          &layout, rank, chunk_count, chunks_per_shard, M, max_comp, 0) == 0);

  // Create fake compressed data: chunk i has size (10 + i) bytes,
  // filled with value i.
  compressed = (char*)calloc(M, max_comp);
  CHECK(Fail, compressed);

  for (uint64_t i = 0; i < M; ++i) {
    comp_sizes[i] = 10 + i;
    memset(compressed + i * max_comp, (int)(i + 1), comp_sizes[i]);
  }

  struct aggregate_cpu_workspace ws;
  memset(&ws, 0, sizeof(ws));
  struct aggregate_result result;
  CHECK(Fail, aggregate_cpu_workspace_init(&ws, &layout) == 0);
  CHECK(Fail, aggregate_cpu_into(compressed, comp_sizes, &layout, &ws, &result) == 0);

  // Verify: each chunk i should appear at its permuted position P[i].
  // The offsets should be a valid prefix sum.
  CHECK(Fail, result.offsets[0] == 0);
  for (uint64_t i = 0; i < layout.covering_count; ++i)
    CHECK(Fail, result.offsets[i + 1] >= result.offsets[i]);

  // Verify data: for each chunk i, find its permuted position and check bytes.
  for (uint64_t i = 0; i < M; ++i) {
    uint32_t pi = (uint32_t)ravel(
      layout.lifted_rank, layout.lifted_shape, layout.lifted_strides, i);
    size_t off = result.offsets[pi];
    size_t sz = result.chunk_sizes[pi];
    CHECK(Fail, sz == comp_sizes[i]);
    const char* p = (const char*)result.data + off;
    for (size_t j = 0; j < sz; ++j)
      CHECK(Fail, (uint8_t)p[j] == (uint8_t)(i + 1));
  }

  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_info("  PASS");
  return 0;

Fail:
  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_error("  FAIL");
  return 1;
}

// Test with multi-shard layout where permutation is non-trivial.
// 3D: chunk_count={4, 3}, chunks_per_shard={2, 3}
// shard_count = {2, 1}, C = 2*2 * 1*3 = 12, M = 12
static int
test_multishard(void)
{
  log_info("=== test_aggregate_cpu_multishard ===");

  struct aggregate_layout layout;
  memset(&layout, 0, sizeof(layout));

  uint64_t chunk_count[] = { 0, 4, 3 };
  uint64_t chunks_per_shard[] = { 0, 2, 3 };
  uint8_t rank = 3;
  uint64_t M = 12;
  size_t max_comp = 64;

  size_t comp_sizes[12];
  char* compressed = NULL;

  CHECK(Fail,
        aggregate_layout_compute(
          &layout, rank, chunk_count, chunks_per_shard, M, max_comp, 0) == 0);

  CHECK(Fail, layout.covering_count == 12);

  compressed = (char*)calloc(M, max_comp);
  CHECK(Fail, compressed);

  for (uint64_t i = 0; i < M; ++i) {
    comp_sizes[i] = 5 + (i % 7);
    memset(compressed + i * max_comp, (int)(i + 1), comp_sizes[i]);
  }

  struct aggregate_cpu_workspace ws;
  memset(&ws, 0, sizeof(ws));
  struct aggregate_result result;
  CHECK(Fail, aggregate_cpu_workspace_init(&ws, &layout) == 0);
  CHECK(Fail, aggregate_cpu_into(compressed, comp_sizes, &layout, &ws, &result) == 0);

  // Verify round-trip: each chunk's data at its permuted offset
  for (uint64_t i = 0; i < M; ++i) {
    uint32_t pi = (uint32_t)ravel(
      layout.lifted_rank, layout.lifted_shape, layout.lifted_strides, i);
    CHECK(Fail, result.chunk_sizes[pi] == comp_sizes[i]);
    const char* p = (const char*)result.data + result.offsets[pi];
    for (size_t j = 0; j < comp_sizes[i]; ++j)
      CHECK(Fail, (uint8_t)p[j] == (uint8_t)(i + 1));
  }

  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_info("  PASS");
  return 0;

Fail:
  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_error("  FAIL");
  return 1;
}

// Test with page_size > 0 (shard alignment padding).
static int
test_page_aligned(void)
{
  log_info("=== test_aggregate_cpu_page_aligned ===");

  struct aggregate_layout layout;
  memset(&layout, 0, sizeof(layout));

  // 3D: chunk_count={2, 2}, chunks_per_shard={1, 2}, shard_count={2, 1}
  // cps_inner = 2, C = 2*1 * 1*2 = 4, M = 4
  uint64_t chunk_count[] = { 0, 2, 2 };
  uint64_t chunks_per_shard[] = { 0, 1, 2 };
  uint8_t rank = 3;
  uint64_t M = 4;
  size_t max_comp = 100;
  size_t page_size = 512;

  size_t comp_sizes[4] = { 30, 40, 50, 60 };
  char* compressed = NULL;

  CHECK(Fail,
        aggregate_layout_compute(
          &layout, rank, chunk_count, chunks_per_shard, M, max_comp, page_size) ==
          0);

  compressed = (char*)calloc(M, max_comp);
  CHECK(Fail, compressed);

  for (uint64_t i = 0; i < M; ++i)
    memset(compressed + i * max_comp, (int)(i + 1), comp_sizes[i]);

  struct aggregate_cpu_workspace ws;
  memset(&ws, 0, sizeof(ws));
  struct aggregate_result result;
  CHECK(Fail, aggregate_cpu_workspace_init(&ws, &layout) == 0);
  CHECK(Fail, aggregate_cpu_into(compressed, comp_sizes, &layout, &ws, &result) == 0);

  // Verify shard boundaries are page-aligned.
  // Each shard has cps_inner chunks. The offset after each shard group
  // should be a multiple of page_size.
  uint64_t cps = layout.cps_inner;
  uint64_t num_shards = layout.covering_count / cps;
  for (uint64_t s = 0; s < num_shards; ++s) {
    size_t shard_end = result.offsets[(s + 1) * cps];
    CHECK(Fail, shard_end % page_size == 0);
  }

  // Verify chunk data integrity (pre-padding sizes).
  for (uint64_t i = 0; i < M; ++i) {
    uint32_t pi = (uint32_t)ravel(
      layout.lifted_rank, layout.lifted_shape, layout.lifted_strides, i);
    CHECK(Fail, result.chunk_sizes[pi] == comp_sizes[i]);
    const char* p = (const char*)result.data + result.offsets[pi];
    for (size_t j = 0; j < comp_sizes[i]; ++j)
      CHECK(Fail, (uint8_t)p[j] == (uint8_t)(i + 1));
  }

  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_info("  PASS");
  return 0;

Fail:
  aggregate_cpu_workspace_free(&ws);
  free(compressed);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_simple();
  rc |= test_multishard();
  rc |= test_page_aligned();
  return rc;
}
