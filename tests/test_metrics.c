// Every per-stage timing number the report prints comes from a pair of CUDA
// events and a ring slot the caller has to mark. Drop the mark and the metric
// keeps count 0, which the report prints as a missing row (#191, #197).
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "util/prelude.h"

#include "test_metric_check.h"
#include "test_runner.h"
#include "test_shard_sink.h"

#include <stdlib.h>

static struct tile_stream_configuration
make_config(struct dimension* dims, const char* downsample_names)
{
  const uint8_t rank = dims_create(dims, "zyx", (uint64_t[]){ 0, 8, 8 });
  dims_set_chunk_sizes(dims, rank, (uint64_t[]){ 2, 2, 2 });
  dims[0].chunks_per_shard = 2;
  dims_set_shard_counts(dims, rank, (uint64_t[]){ 0, 1, 1 });
  dims_set_downsample_by_name(dims, rank, downsample_names);

  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = rank,
    .dimensions = dims,
    .codec = { .id = CODEC_ZSTD },
    .epochs_per_batch = 2,
  };
}

static int
run_stream(const char* downsample_names,
           struct stream_metrics* out,
           struct tile_stream_status* out_status)
{
  // Four batches at two epochs each. The schedule collects the LOD timing ring
  // after every epoch, so it cannot wrap at any number of them.
  const uint64_t epochs = 8;

  struct dimension dims[3];
  struct tile_stream_configuration config = make_config(dims, downsample_names);

  struct test_shard_sink sink;
  const int shards_per_level[] = { 32, 32, 32 };
  test_sink_init_multi(&sink, 3, shards_per_level, 256 * 1024);

  struct tile_stream_gpu* s = NULL;
  uint16_t* src = NULL;
  int ok = 0;

  s = tile_stream_gpu_create(&config, &sink.base);
  CHECK(Fail, s);

  *out_status = tile_stream_gpu_status(s);

  {
    const uint64_t elements =
      epochs * tile_stream_gpu_layout(s)->epoch_elements;
    src = (uint16_t*)malloc(elements * sizeof(uint16_t));
    CHECK(Fail, src);
    // Sequential values keep the codec from collapsing the payload to nothing.
    for (uint64_t i = 0; i < elements; ++i)
      src[i] = (uint16_t)(i & 0xFFFF);

    struct slice input = { .beg = src, .end = src + elements };
    CHECK(Fail, writer_append(tile_stream_gpu_writer(s), input).error == 0);
  }

  CHECK(Fail, writer_flush(tile_stream_gpu_writer(s)).error == 0);

  *out = tile_stream_gpu_get_metrics(s);
  ok = 1;

Fail:
  free(src);
  tile_stream_gpu_destroy(s);
  test_sink_free(&sink);
  return ok ? 0 : 1;
}

// Every stage is checked even after one has failed, so a run that lost several
// reports them all: each rerun here costs a GPU allocation.
static int
check_shared_stages(const struct stream_metrics* m)
{
  int ok = 1;
  ok &= metric_any_arrived_timed(&m->h2d);
  ok &= metric_any_arrived_timed(&m->scatter);
  ok &= metric_any_arrived_timed(&m->compress);
  ok &= metric_any_arrived_timed(&m->aggregate);
  ok &= metric_any_arrived_timed(&m->d2h);
  ok &= metric_any_arrived_timed(&m->sink);
  if (m->tail_gate.count != 0 || m->tail_gate.ms != 0) {
    log_error("  %s: legacy metric is not zero", m->tail_gate.name);
    ok = 0;
  }
  if (m->d2h_logical_payload_bytes == 0 ||
      m->d2h_payload_bytes_transferred != m->d2h_logical_payload_bytes ||
      m->d2h_metadata_bytes_transferred == 0 ||
      m->d2h_payload_copy_count == 0) {
    log_error("  D2H transfer statistics are missing or inconsistent");
    ok = 0;
  }
  for (size_t i = 0; i < sizeof(m->edge_stall) / sizeof(m->edge_stall[0]);
       ++i) {
    if (m->edge_stall[i].wait_calls == 0) {
      log_error("  %s: host dependency was never checked",
                m->edge_stall[i].name);
      ok = 0;
    }
  }
  ok &= no_samples_lost("scatter", m->scatter_samples_lost);
  // The LOD arm cannot fail while the schedule collects every epoch; it guards
  // a change in that cadence rather than proving the collector kept up.
  ok &= no_samples_lost("lod", m->lod_samples_lost);
  return ok;
}

// The pyramid stages run only when there is a pyramid, and the fold only when
// the append dimension is downsampled, so elsewhere their empty rows are the
// truth.
static int
run_case(const char* downsample_names, int expect_pyramid, int expect_fold)
{
  struct stream_metrics m;
  struct tile_stream_status st;
  int ok = 0;

  CHECK(Fail, run_stream(downsample_names, &m, &st) == 0);
  CHECK(Fail, (st.nlod > 1) == expect_pyramid);
  CHECK(Fail, st.append_downsample == expect_fold);

  CHECK(Fail, check_shared_stages(&m));

  if (expect_pyramid) {
    CHECK(Fail, metric_any_arrived_timed(&m.lod_gather));
    CHECK(Fail, metric_any_arrived_timed(&m.lod_reduce));
    CHECK(Fail, metric_any_arrived_timed(&m.lod_morton_chunk));
  } else {
    CHECK(Fail, metric_arrived(&m.lod_gather, 0));
    CHECK(Fail, metric_arrived(&m.lod_reduce, 0));
    CHECK(Fail, metric_arrived(&m.lod_morton_chunk, 0));
  }

  if (expect_fold)
    CHECK(Fail, metric_any_arrived_timed(&m.lod_append_fold));
  else
    CHECK(Fail, metric_arrived(&m.lod_append_fold, 0));

  ok = 1;

Fail:
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

static int
test_metrics_single_level(void)
{
  log_info("=== test_metrics_single_level ===");
  return run_case(NULL, 0, 0);
}

// #191 broke this path: with a pyramid the ingest takes the device-to-device
// copy rather than the scatter, and that copy is the report's Copy row.
static int
test_metrics_multiscale(void)
{
  log_info("=== test_metrics_multiscale ===");
  return run_case("yx", 1, 0);
}

static int
test_metrics_append_downsample(void)
{
  log_info("=== test_metrics_append_downsample ===");
  return run_case("zyx", 1, 1);
}

RUN_GPU_TESTS({ "metrics_single_level", test_metrics_single_level },
              { "metrics_multiscale", test_metrics_multiscale },
              { "metrics_append_downsample", test_metrics_append_downsample }, )
