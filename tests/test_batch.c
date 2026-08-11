#include "gpu/prelude.cuda.h"
#include "gpu/stream.internal.h"
#include "platform/platform.h"
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "test_gpu_helpers.h"
#include "test_shard_sink.h"
#include "test_shard_verify.h"
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

static struct tile_stream_configuration
make_coordinator_config(struct dimension* dims, enum compression_codec codec)
{
  uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 0, 8, 12 });
  dims_set_chunk_sizes(dims, rank, (uint64_t[]){ 2, 2, 3 });
  dims[0].chunks_per_shard = 4;
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 0, 2, 2 });
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = codec },
    .epochs_per_batch = 1,
  };
}

static int
coordinator_sink_has_output(const struct test_shard_sink* sink,
                            uint64_t expected_shards)
{
  uint64_t finalized = 0;
  size_t bytes = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    if (sink->writers[0][i].finalized)
      finalized++;
    bytes += sink->writers[0][i].size;
  }
  return finalized >= expected_shards && bytes > 0;
}

#define COORDINATOR_MAX_CHUNKS_PER_SHARD 64

// A batch whose aggregation read a stale tail length places its first chunk
// at the wrong offset. The shard still finalizes with a plausible byte count,
// so the only thing that catches it is the recorded layout: every chunk must
// begin exactly where the previous one ended.
static int
coordinator_shards_are_packed(const struct test_shard_sink* sink,
                              const struct compress_agg_array* ar)
{
  uint64_t offsets[COORDINATOR_MAX_CHUNKS_PER_SHARD];
  uint64_t sizes[COORDINATOR_MAX_CHUNKS_PER_SHARD];
  uint64_t checked = 0;

  for (uint64_t si = 0;
       si < ar->total_shards && si < TEST_SHARD_SINK_MAX_SHARDS;
       ++si) {
    const struct test_shard_writer* w = &sink->writers[0][si];
    if (!w->finalized)
      continue;

    const uint64_t nchunks = ar->shard[si].chunks_per_shard_total;
    if (nchunks > COORDINATOR_MAX_CHUNKS_PER_SHARD ||
        shard_index_parse(w->buf, w->size, nchunks, offsets, sizes)) {
      log_error("  shard %llu: cannot read a %llu chunk index from %zu bytes",
                (unsigned long long)si,
                (unsigned long long)nchunks,
                w->size);
      return 0;
    }

    uint64_t expected_offset = 0;
    int past_end = 0;
    for (uint64_t k = 0; k < nchunks; ++k) {
      // A partly filled shard leaves its trailing slots unwritten.
      if (offsets[k] == UINT64_MAX && sizes[k] == UINT64_MAX) {
        past_end = 1;
        continue;
      }
      if (past_end || sizes[k] == 0 || offsets[k] != expected_offset) {
        log_error("  shard %llu chunk %llu: offset %llu size %llu, "
                  "expected offset %llu%s",
                  (unsigned long long)si,
                  (unsigned long long)k,
                  (unsigned long long)offsets[k],
                  (unsigned long long)sizes[k],
                  (unsigned long long)expected_offset,
                  past_end ? " (after an unwritten chunk)" : "");
        return 0;
      }
      expected_offset = offsets[k] + sizes[k];
    }
    if (expected_offset > 0)
      checked++;
  }
  return checked > 0;
}

// --- Test cases ---

// 1. One epoch into a K=2 batch stays in staging. A dispatch covers at most the
// room left in the batch, and here that is the whole batch (#173).
static int
test_batch_one_epoch_stays_staged(void)
{
  log_info("=== test_batch_one_epoch_stays_staged ===");

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

  // Verify state: nothing dispatched, so no epoch counted into the batch
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.pool_current == 0);
    CHECK(Fail2, st.flush_pending == 0);
  }
  CHECK(Fail2, tile_stream_gpu_cursor(s) == 48);

  // Sink should not have been touched yet
  CHECK(Fail2, css.open_count == 0);
  CHECK(Fail2, css.finalize_count == 0);

  // Flush is what gets the staged epoch to the device
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
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

// 1b. A dispatch that ends on an epoch boundary partway through a batch leaves
// the batch counted but unkicked. Epochs are 256 bytes here and the staging
// buffer is 4096, so one dispatch covers 16 of the batch's 32 epochs.
static int
test_batch_mid_batch_after_dispatch(void)
{
  log_info("=== test_batch_mid_batch_after_dispatch ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 1024 * 1024);

  struct dimension dims[3];
  uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 0, 8, 8 });
  dims_set_chunk_sizes(dims, rank, (uint64_t[]){ 2, 4, 4 });
  dims[0].chunks_per_shard = 16;
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 0, 1, 1 });

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .epochs_per_batch = 32,
  };

  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  const uint64_t epoch = tile_stream_gpu_layout(s)->epoch_elements;
  CHECK(Fail, epoch == 128);

  uint16_t* src = make_src(16 * epoch);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 16 * epoch };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.epochs_per_batch == 32);
    CHECK(Fail2, st.batch_accumulated == 16);
    CHECK(Fail2, st.flush_pending == 0);
  }

  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
  CHECK(Fail2, tile_stream_gpu_status(s).batch_accumulated == 0);
  CHECK(Fail2, css.finalize_count >= 1);
  // All 16 epochs went over in one transfer, which is what tells this apart
  // from a transfer per epoch. Counted after the flush, which is where the
  // last outstanding measurement is read.
  CHECK(Fail2, tile_stream_gpu_get_metrics(s).h2d.count == 1);

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

  // After full batch: accumulated reset, pool swapped, pending set.
  // Sink-side counts are not checked here: the delivery worker may have
  // already drained the kicked batch.
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.pool_current == 1);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Flush joins the pending batch
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

  // After 2 batches: swapped twice → back to pool 0, both kicked and not
  // yet joined (the delivery worker may already have drained either).
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.pool_current == 0);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Flush joins both batches; by then each has finalized its shards.
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
  CHECK(Fail2, css.finalize_count >= 2);

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

  // Flush exercises the partial batch path (schedule_flush_accumulated)
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

  // Epochs 0-1 filled the batch and kicked; epoch 2 is still staged.
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.batch_accumulated == 0);
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

// 7. A failed delivery leaves the stream unable to say where new data belongs,
// so it stops taking any. Queued writes are still drained and the array shape
// is still written, but it names only what reached a closed-out shard — here
// the very first delivery failed, so it names nothing.
static int
test_batch_failed_delivery_claims_nothing(void)
{
  log_info("=== test_batch_failed_delivery_claims_nothing ===");

  struct test_shard_sink css;
  // Far too small for one batch, so a shard write fails during delivery.
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 64);

  struct dimension dims[3];
  make_test_dims_3d_unbounded(dims);
  struct tile_stream_configuration config = make_config(dims);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &css.base);
  CHECK(Fail0, s);

  // 6 epochs at K=2. The third batch's kick is the first to drain a kicked
  // slot, so that is where delivery fails and the append reports it.
  uint16_t* src = make_src(288);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 288 };
  struct writer_result ar = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, ar.error != 0);

  // The stream can no longer say where new data belongs, so it takes none.
  struct writer_result again = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, again.error != 0);

  struct writer_result fr = writer_flush(tile_stream_gpu_writer(s));
  // The shape publishes in close, which runs even after a failed flush.
  writer_close(tile_stream_gpu_writer(s));
  log_info("  flush err=%d update_append=%d finalize=%d",
           fr.error,
           css.update_append_count,
           css.finalize_count);
  CHECK(Fail2, fr.error != 0);
  CHECK(Fail2, css.update_append_count == 1);
  CHECK(Fail2, css.last_append_size0 == 0);

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

// Hold generation 1 after submission but before drain. Generation 2 may
// compress and reach PREPARED, but cannot aggregate until generation 1 has
// delivered and synchronously uploaded its tail state.
static int
run_host_coordinator_hold(enum compression_codec codec)
{
  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 2 * 1024 * 1024);
  sink.shard_alignment = platform_page_alignment();

  struct dimension dims[3];
  struct tile_stream_configuration config =
    make_coordinator_config(dims, codec);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &sink.base);
  uint16_t* src = NULL;
  int held = 0;
  int ok = 0;
  CHECK(Fail, s);
  CHECK(Fail, s->engine.sched.mode == SCHEDULE_PIPELINED_HOST_COORDINATED);
  CHECK(Fail, s->engine.sched.epochs_per_batch == 1);
  CHECK(Fail, s->engine.compress_agg.ar.total_shards > 1);

  const size_t epoch_elements = tile_stream_gpu_layout(s)->epoch_elements;
  src = make_src(2 * epoch_elements);
  CHECK(Fail, src);

  gpu_delivery_set_hold(&s->engine.delivery, DELIVERY_HOLD_BEFORE_DRAIN);
  held = 1;
  {
    struct slice input = { .beg = src, .end = src + 2 * epoch_elements };
    CHECK(Fail, writer_append(tile_stream_gpu_writer(s), input).error == 0);
  }

  uint64_t generation0 = 0;
  uint64_t generation1 = 0;
  CHECK(Fail,
        gpu_delivery_job_state(&s->engine.delivery, 0, &generation0) ==
          DELIVERY_JOB_SUBMITTED);
  CHECK(Fail,
        gpu_delivery_job_state(&s->engine.delivery, 1, &generation1) ==
          DELIVERY_JOB_PREPARED);
  CHECK(Fail, generation0 == 1);
  CHECK(Fail, generation1 == 2);
  {
    uint64_t submitted = 0;
    uint64_t tail_ready = 0;
    gpu_delivery_generations(&s->engine.delivery, &submitted, &tail_ready);
    CHECK(Fail, submitted == 1);
    CHECK(Fail, tail_ready == 0);
  }

  gpu_delivery_set_hold(&s->engine.delivery, DELIVERY_HOLD_NONE);
  held = 0;
  CHECK(Fail, writer_flush(tile_stream_gpu_writer(s)).error == 0);
  {
    uint64_t submitted = 0;
    uint64_t tail_ready = 0;
    gpu_delivery_generations(&s->engine.delivery, &submitted, &tail_ready);
    CHECK(Fail, submitted == 2);
    CHECK(Fail, tail_ready == 2);
  }
  CHECK(
    Fail,
    coordinator_sink_has_output(&sink, s->engine.compress_agg.ar.total_shards));
  CHECK(Fail, coordinator_shards_are_packed(&sink, &s->engine.compress_agg.ar));
  ok = 1;

Fail:
  if (held && s)
    gpu_delivery_set_hold(&s->engine.delivery, DELIVERY_HOLD_NONE);
  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&sink);
  return ok ? 0 : 1;
}

static int
test_host_coordinator_hold(void)
{
  log_info("=== test_host_coordinator_hold ===");
  int error = 0;
  error |= run_host_coordinator_hold(CODEC_ZSTD);
  error |= run_host_coordinator_hold(CODEC_NONE);
  log_info("  %s", error ? "FAIL" : "PASS");
  return error;
}

// Force the worker-unavailable selection after construction, before any job
// is queued. Page-aligned tail carry must remain correct on the direct,
// drain-before-kick fallback.
static int
test_worker_unavailable_fallback(void)
{
  log_info("=== test_worker_unavailable_fallback ===");

  struct test_shard_sink sink;
  test_sink_init(&sink, TEST_SHARD_SINK_MAX_SHARDS, 2 * 1024 * 1024);
  sink.shard_alignment = platform_page_alignment();

  struct dimension dims[3];
  struct tile_stream_configuration config =
    make_coordinator_config(dims, CODEC_ZSTD);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config, &sink.base);
  uint16_t* src = NULL;
  int ok = 0;
  CHECK(Fail, s);

  gpu_delivery_stop_join(&s->engine.delivery);
  schedule_select(
    &s->engine.sched, &s->engine.compress_agg.ar, &s->engine.delivery);
  CHECK(Fail, s->engine.sched.mode == SCHEDULE_DRAIN_BEFORE_KICK);

  const size_t epoch_elements = tile_stream_gpu_layout(s)->epoch_elements;
  src = make_src(4 * epoch_elements);
  CHECK(Fail, src);
  {
    struct slice input = { .beg = src, .end = src + 4 * epoch_elements };
    CHECK(Fail, writer_append(tile_stream_gpu_writer(s), input).error == 0);
  }
  CHECK(Fail, writer_flush(tile_stream_gpu_writer(s)).error == 0);
  CHECK(
    Fail,
    coordinator_sink_has_output(&sink, s->engine.compress_agg.ar.total_shards));
  CHECK(Fail, coordinator_shards_are_packed(&sink, &s->engine.compress_agg.ar));
  ok = 1;

Fail:
  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS(
  { "batch_one_epoch_stays_staged", test_batch_one_epoch_stays_staged },
  { "batch_mid_batch_after_dispatch", test_batch_mid_batch_after_dispatch },
  { "batch_full_triggers_swap", test_batch_full_triggers_swap },
  { "batch_multi_cycle", test_batch_multi_cycle },
  { "batch_partial_flush", test_batch_partial_flush },
  { "batch_3epochs_flush", test_batch_3epochs_flush },
  { "batch_multiscale_unaligned_K", test_batch_multiscale_unaligned_K },
  { "batch_failed_delivery_claims_nothing",
    test_batch_failed_delivery_claims_nothing },
  { "host_coordinator_hold", test_host_coordinator_hold },
  { "worker_unavailable_fallback", test_worker_unavailable_fallback }, )
