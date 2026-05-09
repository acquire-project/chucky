// Contract test for the CPU-side aggregate_result.
//
// Drives aggregate_cpu_batch_into_unified with synthetic compressed input,
// then runs the shared verifier from aggregate_contract.h. Three scenarios:
//   1. carry-over, no leading tail (first batch).
//   2. carry-over, with leading tail on each shard (subsequent batch).
//   3. legacy contiguous (page_size == 0).

#include "aggregate_contract.h"

#include "cpu/aggregate.h"
#include "lod/lod_plan.h"
#include "stream/types.aggregate.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"
#include "zarr/shard_delivery.h"

#include <stdlib.h>
#include <string.h>

static struct threadpool* g_pool;

// One LOD, deterministic geometry. n_append = 1 => append dim is "t".
//   t = unbounded, chunk=1, cps=2  (cps_append=2)
//   y = 4,         chunk=2, cps=2
//   x = 8,         chunk=2, cps=2  (=> 4 x-chunks; 2 along x => 2 inner shards)
//
// Per-epoch chunks (inner): y_chunks * x_chunks = 2 * 4 = 8.
// cps_inner = cps_y * cps_x = 4. num_shards = 8 / 4 = 2.
// Each shard: cps_inner=4 chunks per epoch.
struct geom
{
  uint8_t rank;
  uint8_t n_append;
  uint64_t chunk_count[3]; // {append placeholder, y, x}
  uint64_t chunks_per_shard[3];
  uint64_t M; // chunks per epoch (inner only)
  size_t max_comp;
  uint32_t batch_count;
  uint64_t cps_append;
};

static struct geom
make_geom(void)
{
  return (struct geom){
    .rank = 3,
    .n_append = 1,
    .chunk_count = { 0, 2, 4 },
    .chunks_per_shard = { 0, 2, 2 },
    .M = 8,
    .max_comp = 64,
    .batch_count = 1,
    .cps_append = 2,
  };
}

// Compute per-LOD aggregate_layout for our nlod=1 geometry.
static int
build_layout(struct aggregate_layout* out,
             const struct geom* g,
             size_t page_size)
{
  return aggregate_layout_compute(out,
                                  g->rank,
                                  g->n_append,
                                  g->chunk_count,
                                  g->chunks_per_shard,
                                  g->M,
                                  g->max_comp,
                                  g->batch_count,
                                  page_size,
                                  g->cps_append);
}

// Run aggregate_cpu_batch_into_unified for one LOD, fully synthesized.
//
// `tail_in` (length num_shards) is OPTIONAL: when non-NULL the per-shard
// shard_state.tail_buf entries are populated with that many bytes (filled
// with a recognizable pattern), so we can later assert delivery's
// leading-tail copy lands them at the head of each shard's region.
//
// Returns 0 on success and writes the per-LOD result into *out_result.
static int
run_aggregate(const struct geom* g,
              size_t page_size,
              const size_t* tail_in,
              struct aggregate_layout* out_layout,
              struct aggregate_cpu_workspace* out_ws,
              struct aggregate_result* out_result,
              struct shard_state* out_shard,
              uint8_t** out_compressed,
              size_t** out_comp_sizes,
              uint32_t** out_gather_perm)
{
  CHECK(Fail, build_layout(out_layout, g, page_size) == 0);

  const uint64_t covering = out_layout->covering_count;
  const uint64_t M = out_layout->chunks_per_epoch;
  const uint64_t num_shards = out_layout->num_shards;
  const uint32_t n_active = g->batch_count;

  // Per-LOD layouts (length nlod=1).
  struct aggregate_layout per_lod[1];
  per_lod[0] = *out_layout;

  uint32_t per_lod_n_active[1] = { n_active };
  struct batch_aggregate_layout blayout;
  CHECK(Fail,
        batch_aggregate_layout_init(
          &blayout, per_lod, per_lod_n_active, 1, page_size) == 0);

  // Workspace sized for one LOD, one batch.
  // (total_batch_covering + nlod) accounts for the lv-shift slack.
  out_ws->perm =
    (uint32_t*)calloc(blayout.total_batch_chunks, sizeof(uint32_t));
  out_ws->permuted_sizes =
    (size_t*)calloc(blayout.total_batch_covering + 1, sizeof(size_t));
  out_ws->offsets =
    (size_t*)calloc(blayout.total_batch_covering + 2, sizeof(size_t));
  out_ws->chunk_sizes =
    (size_t*)calloc(blayout.total_batch_covering + 1, sizeof(size_t));
  out_ws->data_capacity = blayout.total_data_bytes;
  out_ws->data = calloc(1, out_ws->data_capacity);
  CHECK(Fail,
        out_ws->perm && out_ws->permuted_sizes && out_ws->offsets &&
          out_ws->chunk_sizes && out_ws->data);

  // Build perm/gather LUTs for nlod=1, single batch (pool epoch 0).
  uint32_t pool_epochs[1] = { 0 };
  struct level_geometry levels = { 0 };
  levels.nlod = 1;
  levels.total_chunks = M;
  levels.level[0].chunk_count = M;
  levels.level[0].chunk_offset = 0;

  *out_gather_perm =
    (uint32_t*)calloc(blayout.total_batch_chunks * 2, sizeof(uint32_t));
  CHECK(Fail, *out_gather_perm);
  uint32_t* gather = *out_gather_perm;
  uint32_t* perm = *out_gather_perm + blayout.total_batch_chunks;
  aggregate_batch_luts(
    out_layout, &levels, /*lv=*/0, n_active, pool_epochs, gather, perm);
  // For nlod=1 the +lv shift is +0; perm targets already index the LOD's
  // (only) slice of ws->offsets correctly.
  memcpy(out_ws->perm, perm, blayout.total_batch_chunks * sizeof(uint32_t));

  // Synthetic compressed input: chunk i has size (10 + i) bytes filled with
  // value (i + 1) to make off-by-one errors visible.
  *out_comp_sizes = (size_t*)calloc(M, sizeof(size_t));
  *out_compressed = (uint8_t*)calloc(M, g->max_comp);
  CHECK(Fail, *out_comp_sizes && *out_compressed);
  for (uint64_t i = 0; i < M; ++i) {
    (*out_comp_sizes)[i] = 10 + (i % 7);
    memset(*out_compressed + i * g->max_comp,
           (int)((i + 1) & 0xff),
           (*out_comp_sizes)[i]);
  }

  // Shard state with synthetic tail bytes.
  memset(out_shard, 0, sizeof(*out_shard));
  struct level_layout_info li = {
    .agg_layout = *out_layout,
    .batch_active_count = n_active,
    .chunks_per_shard_append = g->cps_append,
    .chunks_per_shard_inner = out_layout->cps_inner,
    .chunks_per_shard_total = g->cps_append * out_layout->cps_inner,
    .shard_inner_count = num_shards,
  };
  CHECK(Fail, init_shard_state(out_shard, &li) == 0);

  // Pre-load shard_state with caller-supplied tail bytes.
  if (tail_in && page_size > 0) {
    for (uint64_t si = 0; si < num_shards; ++si) {
      const size_t n = tail_in[si];
      out_shard->shards[si].tail_bytes = n;
      if (n > 0)
        memset(out_shard->shards[si].tail_buf, 0xAB, n);
    }
  }

  size_t* h_tail_array = (size_t*)calloc(num_shards, sizeof(size_t));
  CHECK(Fail, h_tail_array);
  if (tail_in && page_size > 0)
    memcpy(h_tail_array, tail_in, num_shards * sizeof(size_t));
  size_t* h_tail_per_lod[1] = { h_tail_array };

  struct shard_state* shards_per_lod[1] = { out_shard };
  struct aggregate_result results[1];

  CHECK(Fail,
        aggregate_cpu_batch_into_unified(&(struct aggregate_cpu_inputs){
          .compressed_base = *out_compressed,
          .comp_sizes_base = *out_comp_sizes,
          .gather = gather,
          .layout = &blayout,
          .per_lod_layouts = per_lod,
          .shards_by_lod = page_size > 0 ? shards_per_lod : NULL,
          .h_tail_bytes = page_size > 0 ? h_tail_per_lod : NULL,
          .ws = out_ws,
          .per_lod_results = results,
          .pool = g_pool,
        }) == 0);

  *out_result = results[0];
  free(h_tail_array);
  // covering and M referenced in test logic via the layout; suppress unused.
  (void)covering;
  return 0;

Fail:
  return 1;
}

static void
free_run(struct aggregate_layout* layout,
         struct aggregate_cpu_workspace* ws,
         struct shard_state* shard,
         uint8_t* compressed,
         size_t* comp_sizes,
         uint32_t* gather_perm)
{
  shard_state_destroy(shard);
  free(ws->perm);
  free(ws->permuted_sizes);
  free(ws->offsets);
  free(ws->chunk_sizes);
  free(ws->data);
  free(compressed);
  free(comp_sizes);
  free(gather_perm);
  // aggregate_layout has no host-allocated members (the d_* device pointers
  // stay NULL on the CPU path); nothing to free here.
  (void)layout;
  memset(ws, 0, sizeof(*ws));
}

static int
test_carryover_no_tail(void)
{
  log_info("=== test_carryover_no_tail ===");
  struct geom g = make_geom();
  const size_t page_size = 4096;

  struct aggregate_layout layout = { 0 };
  struct aggregate_cpu_workspace ws = { 0 };
  struct aggregate_result result = { 0 };
  struct shard_state shard = { 0 };
  uint8_t* compressed = NULL;
  size_t* comp_sizes = NULL;
  uint32_t* gather_perm = NULL;

  CHECK(Fail,
        run_aggregate(&g,
                      page_size,
                      NULL, // no leading tails (first batch)
                      &layout,
                      &ws,
                      &result,
                      &shard,
                      &compressed,
                      &comp_sizes,
                      &gather_perm) == 0);

  CHECK(Fail,
        verify_aggregate_result_carryover(
          &result, &layout, NULL, g.batch_count) == 0);

  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_info("  PASS");
  return 0;
Fail:
  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_error("  FAIL");
  return 1;
}

static int
test_carryover_with_tail(void)
{
  log_info("=== test_carryover_with_tail ===");
  struct geom g = make_geom();
  const size_t page_size = 4096;
  // Pretend each shard carried different non-zero sub-page tails from the
  // prior batch. The verifier asserts each shard's first chunk lands at
  // si*shard_capacity + tail_in[si].
  const size_t tail_in[2] = { 17, 113 };

  struct aggregate_layout layout = { 0 };
  struct aggregate_cpu_workspace ws = { 0 };
  struct aggregate_result result = { 0 };
  struct shard_state shard = { 0 };
  uint8_t* compressed = NULL;
  size_t* comp_sizes = NULL;
  uint32_t* gather_perm = NULL;

  CHECK(Fail,
        run_aggregate(&g,
                      page_size,
                      tail_in,
                      &layout,
                      &ws,
                      &result,
                      &shard,
                      &compressed,
                      &comp_sizes,
                      &gather_perm) == 0);

  CHECK(Fail,
        verify_aggregate_result_carryover(
          &result, &layout, tail_in, g.batch_count) == 0);

  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_info("  PASS");
  return 0;
Fail:
  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_error("  FAIL");
  return 1;
}

static int
test_contiguous(void)
{
  log_info("=== test_contiguous ===");
  struct geom g = make_geom();
  const size_t page_size = 0;

  struct aggregate_layout layout = { 0 };
  struct aggregate_cpu_workspace ws = { 0 };
  struct aggregate_result result = { 0 };
  struct shard_state shard = { 0 };
  uint8_t* compressed = NULL;
  size_t* comp_sizes = NULL;
  uint32_t* gather_perm = NULL;

  CHECK(Fail,
        run_aggregate(&g,
                      page_size,
                      NULL,
                      &layout,
                      &ws,
                      &result,
                      &shard,
                      &compressed,
                      &comp_sizes,
                      &gather_perm) == 0);

  CHECK(Fail,
        verify_aggregate_result_contiguous(&result, &layout, g.batch_count) ==
          0);

  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_info("  PASS");
  return 0;
Fail:
  free_run(&layout, &ws, &shard, compressed, comp_sizes, gather_perm);
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  g_pool = threadpool_new(1);
  if (!g_pool)
    return 1;

  int rc = 0;
  rc |= test_carryover_no_tail();
  rc |= test_carryover_with_tail();
  rc |= test_contiguous();

  threadpool_free(g_pool);
  return rc;
}
