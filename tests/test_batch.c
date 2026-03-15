#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_shard_sink.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Config builder ---
// Rank=3, dims {0, 4, 6}, tiles {2, 2, 3}, tiles_per_shard {2, 2, 2}.
// Unbounded dim0 (size=0). Explicit epochs_per_batch=2.
// Derived: tile_elements=12, tiles_per_epoch=4, epoch_elements=48.

static struct tile_stream_configuration
make_config(const struct dimension* dims, struct shard_sink* sink)
{
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 48 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .shard_sink = sink,
    .codec = CODEC_NONE,
    .epochs_per_batch = 2,
  };
}

static const struct dimension test_dims[] = {
  { .size = 0, .tile_size = 2, .tiles_per_shard = 2, .storage_position = 0 },
  { .size = 4, .tile_size = 2, .tiles_per_shard = 2, .storage_position = 1 },
  { .size = 6, .tile_size = 3, .tiles_per_shard = 2, .storage_position = 2 },
};

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

  struct tile_stream_configuration config = make_config(test_dims, &css.base);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config);
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

  struct tile_stream_configuration config = make_config(test_dims, &css.base);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config);
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

  struct tile_stream_configuration config = make_config(test_dims, &css.base);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config);
  CHECK(Fail0, s);

  // Feed 192 elements (4 epochs = 2 batches)
  uint16_t* src = make_src(192);
  CHECK(Fail, src);

  struct slice input = { .beg = src, .end = src + 192 };
  struct writer_result r = writer_append(tile_stream_gpu_writer(s), input);
  CHECK(Fail2, r.error == 0);

  // After 2 batches: swapped twice → back to pool 0, batch 2 pending
  {
    struct tile_stream_status st = tile_stream_gpu_status(s);
    CHECK(Fail2, st.pool_current == 0);
    CHECK(Fail2, st.batch_accumulated == 0);
    CHECK(Fail2, st.flush_pending == 1);
  }

  // Batch 1 drained when batch 2 started; batch 2 still pending
  int pre_flush_finalize = css.finalize_count;
  CHECK(Fail2, pre_flush_finalize >= 1);

  // Flush drains batch 2
  r = writer_flush(tile_stream_gpu_writer(s));
  CHECK(Fail2, r.error == 0);
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

// 4. Partial batch via explicit flush: 1 epoch then flush.
static int
test_batch_partial_flush(void)
{
  log_info("=== test_batch_partial_flush ===");

  struct test_shard_sink css;
  test_sink_init(&css, TEST_SHARD_SINK_MAX_SHARDS, 512 * 1024);

  struct tile_stream_configuration config = make_config(test_dims, &css.base);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config);
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

  struct tile_stream_configuration config = make_config(test_dims, &css.base);
  struct tile_stream_gpu* s = tile_stream_gpu_create(&config);
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

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_batch_counter_one_epoch();
  log_info("");
  ecode |= test_batch_full_triggers_swap();
  log_info("");
  ecode |= test_batch_multi_cycle();
  log_info("");
  ecode |= test_batch_partial_flush();
  log_info("");
  ecode |= test_batch_3epochs_flush();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
