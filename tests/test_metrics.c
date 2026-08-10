// Every per-stage timing number the report prints comes from a pair of CUDA
// events and a ring slot the caller has to mark. Drop the mark and the metric
// keeps count 0, which the report prints as a missing row rather than a zero
// (#191, #197). These tests run a whole stream and check that each stage the
// run actually exercised reported at least one measurement.
#include "stream.gpu.h"
#include "stream/layouts.h"
#include "util/prelude.h"

#include "test_metric_check.h"
#include "test_runner.h"
#include "test_shard_sink.h"

#include <stdlib.h>

#define EPOCHS 8

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

// Append EPOCHS epochs and flush, then hand back what the stream measured.
static int
run_stream(const char* downsample_names,
           struct stream_metrics* out,
           struct tile_stream_status* out_status)
{
  struct dimension dims[3];
  struct tile_stream_configuration config = make_config(dims, downsample_names);

  struct test_shard_sink sink;
  const int shards_per_level[] = { 32, 32, 32 };
  test_sink_init_multi(&sink, 3, shards_per_level, 4 * 1024 * 1024);

  struct tile_stream_gpu* s = NULL;
  uint16_t* src = NULL;
  int ok = 0;

  s = tile_stream_gpu_create(&config, &sink.base);
  CHECK(Fail, s);

  *out_status = tile_stream_gpu_status(s);

  {
    const uint64_t elements =
      EPOCHS * tile_stream_gpu_layout(s)->epoch_elements;
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

// Stages every run goes through, whatever the level geometry is.
static int
check_shared_stages(const struct stream_metrics* m)
{
  return metric_arrived_at_least_once(&m->h2d) &&
         metric_arrived_at_least_once(&m->scatter) &&
         metric_arrived_at_least_once(&m->compress) &&
         metric_arrived_at_least_once(&m->aggregate) &&
         metric_arrived_at_least_once(&m->d2h) &&
         metric_arrived_at_least_once(&m->sink) &&
         // The gate is a wait, and a wait that never had to wait measures
         // zero, so only its count says whether it was read.
         m->tail_gate.count > 0;
}

static int
check_nothing_lost(const struct stream_metrics* m)
{
  if (m->scatter_samples_lost != 0)
    log_error("  scatter ring wrapped %llu times",
              (unsigned long long)m->scatter_samples_lost);
  if (m->lod_samples_lost != 0)
    log_error("  lod ring wrapped %llu times",
              (unsigned long long)m->lod_samples_lost);
  return m->scatter_samples_lost == 0 && m->lod_samples_lost == 0;
}

// A single-level stream: the LOD stages never run, so their empty rows are
// the truth rather than a lost measurement.
static int
test_metrics_single_level(void)
{
  log_info("=== test_metrics_single_level ===");

  struct stream_metrics m;
  struct tile_stream_status st;
  int ok = 0;

  CHECK(Fail, run_stream(NULL, &m, &st) == 0);
  CHECK(Fail, st.nlod == 1);

  CHECK(Fail, check_shared_stages(&m));
  CHECK(Fail, check_nothing_lost(&m));

  CHECK(Fail, m.lod_gather.count == 0);
  CHECK(Fail, m.lod_reduce.count == 0);
  CHECK(Fail, m.lod_append_fold.count == 0);
  CHECK(Fail, m.lod_morton_chunk.count == 0);

  ok = 1;

Fail:
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Downsampling the two bounded dimensions builds a pyramid without folding
// along the append dimension. This is the path #191 broke: it takes the
// device-to-device copy rather than the scatter, and the copy is what the
// report calls Copy once multiscale is on.
static int
test_metrics_multiscale(void)
{
  log_info("=== test_metrics_multiscale ===");

  struct stream_metrics m;
  struct tile_stream_status st;
  int ok = 0;

  CHECK(Fail, run_stream("yx", &m, &st) == 0);
  CHECK(Fail, st.nlod > 1);
  CHECK(Fail, st.append_downsample == 0);

  CHECK(Fail, check_shared_stages(&m));
  CHECK(Fail, check_nothing_lost(&m));

  CHECK(Fail, metric_arrived_at_least_once(&m.lod_gather));
  CHECK(Fail, metric_arrived_at_least_once(&m.lod_reduce));
  CHECK(Fail, metric_arrived_at_least_once(&m.lod_morton_chunk));
  // The fold runs only when the append dimension is downsampled too.
  CHECK(Fail, m.lod_append_fold.count == 0);

  ok = 1;

Fail:
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Downsampling the append dimension as well adds the fold, the one LOD stage
// the previous case cannot reach.
static int
test_metrics_append_downsample(void)
{
  log_info("=== test_metrics_append_downsample ===");

  struct stream_metrics m;
  struct tile_stream_status st;
  int ok = 0;

  CHECK(Fail, run_stream("zyx", &m, &st) == 0);
  CHECK(Fail, st.nlod > 1);
  CHECK(Fail, st.append_downsample == 1);

  CHECK(Fail, check_shared_stages(&m));
  CHECK(Fail, check_nothing_lost(&m));

  CHECK(Fail, metric_arrived_at_least_once(&m.lod_gather));
  CHECK(Fail, metric_arrived_at_least_once(&m.lod_reduce));
  CHECK(Fail, metric_arrived_at_least_once(&m.lod_morton_chunk));
  CHECK(Fail, metric_arrived_at_least_once(&m.lod_append_fold));

  ok = 1;

Fail:
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

RUN_GPU_TESTS({ "metrics_single_level", test_metrics_single_level },
              { "metrics_multiscale", test_metrics_multiscale },
              { "metrics_append_downsample", test_metrics_append_downsample }, )
