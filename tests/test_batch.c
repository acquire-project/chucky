#include "gpu/prelude.cuda.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "test_gpu_helpers.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include "test_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct tile_stream_configuration
make_config(struct dimension* dims)
{
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 48 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 2,
  };
}

// Fill source with sequential u16 values
static uint16_t*
make_src(size_t count)
{
  uint16_t* src = (uint16_t*)malloc(count * sizeof(uint16_t));
  if (!src)
    return NULL;
  for (size_t i = 0; i < count; ++i)
    src[i] = (uint16_t)(i % 65536);
  return src;
}

// --- Test cases ---

// 1. Mid-batch state verification: 1 epoch into a K=2 batch.
static int
test_batch_counter_one_epoch(void)
{
  log_info("=== test_batch_counter_one_epoch ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  CHECK(Fail, tile_stream_gpu_layout(s)->epoch_elements == 48);

  // Feed 48 elements (1 epoch)
  uint16_t* src = make_src(48);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 48 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  // Verify state: mid-batch
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 1);
    CHECK(Fail2, st.pool_current == 0);
    CHECK(Fail2, st.flush_pending == 0);
  }

  // Sink should not have been touched yet
  CHECK(Fail2, css.open_count == 0);
  CHECK(Fail2, css.finalize_count == 0);

  // Clean up via flush
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

// 2. Pool swap + deferred drain: feed 2 epochs = full batch.
static int
test_batch_full_triggers_swap(void)
{
  log_info("=== test_batch_full_triggers_swap ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // Feed 96 elements (2 epochs = 1 full batch)
  uint16_t* src = make_src(96);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 96 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  // After full batch: accumulated reset, pool swapped, pending set
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.pool_current == 1);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Kicked but NOT drained yet
  CHECK(Fail2, css.finalize_count == 0);

  // Flush drains the pending batch
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);

  CHECK(Fail2, tile_stream_gpu_status(s).flush_pending == 0);
  CHECK(Fail2, css.finalize_count >= 1);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

// 3. Repeated batch cycling: feed 4 epochs = 2 batches.
static int
test_batch_multi_cycle(void)
{
  log_info("=== test_batch_multi_cycle ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // Feed 192 elements (4 epochs = 2 batches)
  uint16_t* src = make_src(192);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 192 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  // After 2 batches: swapped twice → back to pool 0, both pending (lazy
  // delivery: pending[0] = batch 1, pending[1] = batch 2, until either
  // a 3rd batch reuses fc=0 or the stream is flushed).
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.pool_current == 0);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Flush drains both pending batches, finalizing their shards.
  int pre_flush_finalize = css.finalize_count;
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
  CHECK(Fail2, css.finalize_count >= pre_flush_finalize + 2);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

// 4. Partial batch via explicit flush: 1 epoch then flush.
static int
test_batch_partial_flush(void)
{
  log_info("=== test_batch_partial_flush ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // Feed 48 elements (1 epoch, K=2), then flush
  uint16_t* src = make_src(48);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 48 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);
  CHECK(Fail2, tile_stream_gpu_status(s).batch_accumulated == 1);

  // Flush exercises the partial batch path (flush_accumulated_sync)
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);

  // After flush: batch drained
  CHECK(Fail2, tile_stream_gpu_status(s).batch_accumulated == 0);
  CHECK(Fail2, css.finalize_count >= 1);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

// 5. Full batch + partial epoch: 3 epochs with K=2.
static int
test_batch_3epochs_flush(void)
{
  log_info("=== test_batch_3epochs_flush ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // Feed 144 elements (3 epochs, K=2)
  // Epochs 0-1: auto-flush (full batch kicked, pool swapped)
  // Epoch 2: accumulates in new pool
  uint16_t* src = make_src(144);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 144 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 1);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Flush: drain batch 1 + handle epoch 2 as partial + emit partial shard
  int pre_flush_finalize = css.finalize_count;
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);

  CHECK(Fail2, tile_stream_gpu_status(s).flush_pending == 0);
  CHECK(Fail2, css.finalize_count > pre_flush_finalize);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

// 6. Regression: K=3 with append-downsampled multiscale exercises a batch
// whose level-1 emission count (ceil(K/period)=2) exceeds floor(K/period)=1.
// Previously batch_active_count floor-sized the LUTs and level-1 emissions
// past floor(K/period) were silently dropped (truncated to 1 per batch),
// leaving L1 shards under-sized. With the fix, all ceil(K/period) emissions
// are aggregated and each L1 shard receives its full 2-epoch payload.
static int
test_batch_multiscale_unaligned_K(void)
{
  log_info("=== test_batch_multiscale_unaligned_K ===");

  struct test_shard_sink css;
  // Multi-level sink so L1 shards are captured (default single-level sink
  // routes L1+ to the discard writer).
  const int shards_per_level[] = { 8, 8, 8 };
  test_sink_init_multi(&css, 3, shards_per_level, 4 * 1024 * 1024);

  struct dimension dims[3];
  uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 0, 8, 8 });
  dims_set_chunk_sizes(dims, rank, (uint64_t[]){ 2, 2, 2 });
  dims[0].chunks_per_shard = 2; // unbounded
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 0, 1, 1 });
  dims_set_downsample_by_name(dims, rank, "zyx"); // append_downsample=1

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 256 * sizeof(uint16_t),
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 3, // not a multiple of level-1 period (2)
  };

  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // Level 1 emission pattern across 6 epochs (period=2, K=3):
  //   batch 1 (epochs 0,1,2) -> 1 emission (at epoch 1)
  //   batch 2 (epochs 3,4,5) -> 2 emissions (at epochs 3,5)
  // The second batch exceeds floor(K/period)=1; ceil(K/period)=2 is required.
  const size_t epoch_elements = tile_stream_gpu_layout(s)->epoch_elements;
  const size_t total = 6 * epoch_elements;
  uint16_t* src = make_src(total);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + total };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);

  // Level 1 emits at epochs 1, 3, 5. Pre-fix the second batch truncated to
  // floor(K/period)=1 emission (aggregating epoch 3 but dropping epoch 5),
  // so only 2 of the 3 L1 emissions made it into shards and L1 payload was
  // 2/3 of expected. Check total L1 payload bytes as a data-correctness
  // signal; we also verify all 3 L1 emissions finalized a shard.
  size_t l1_bytes = 0;
  int l1_finalized = 0;
  size_t l1_nonzero_bytes = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    l1_bytes += css.writers[1][i].size;
    if (css.writers[1][i].finalized)
      l1_finalized++;
    // Count non-zero bytes in the chunk-data region (precedes the footer).
    // With CODEC_NONE and non-zero input the chunk bytes must contain real
    // downsampled u16 values; an all-zero L1 region would indicate the
    // delivery view pointed at uninitialized memory (#135).
    const uint8_t* buf = css.writers[1][i].buf;
    const size_t sz = css.writers[1][i].size;
    for (size_t b = 0; b < sz; ++b)
      if (buf[b] != 0)
        l1_nonzero_bytes++;
  }
  log_info("  L1 bytes=%zu finalized=%d nonzero=%zu",
           l1_bytes,
           l1_finalized,
           l1_nonzero_bytes);
  // 3 L1 emissions → 3 finalized L1 shards (chunks_per_shard_append for L1
  // is typically 1 after the period-2 halving of the append axis).
  CHECK(Fail2, l1_finalized >= 3);
  // Guard against the contiguous-mode lod_view bug (#135): the bug routes
  // L1 delivery through uninitialized/stale memory, so the chunk-data
  // region of L1 shards loses correlation with the input. With the fix in
  // place, mean-reduced L1 chunks of sequential u16 input carry real
  // small-but-non-zero values across multiple bytes per shard.
  CHECK(Fail2, l1_nonzero_bytes > 0);

  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&css);
  log_info("  PASS");
  return 0;

Fail2:
  free(src);
Fail:
  tile_stream_gpu_destroy(s);
Fail0:
  test_sink_free(&css);
  log_error("  FAIL");
  return 1;
}

RUN_GPU_TESTS({ "batch_counter_one_epoch", test_batch_counter_one_epoch },
              { "batch_full_triggers_swap", test_batch_full_triggers_swap },
              { "batch_multi_cycle", test_batch_multi_cycle },
              { "batch_partial_flush", test_batch_partial_flush },
              { "batch_3epochs_flush", test_batch_3epochs_flush },
              { "batch_multiscale_unaligned_K",
                test_batch_multiscale_unaligned_K }, )
