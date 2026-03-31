#include "dimension.h"
#include "multiarray.cpu.h"
#include "stream.cpu.h"
#include "test_shard_sink.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define SHARD_CAP (1 << 20)
#define SINK_N_SHARDS 16
#define test_sink_init_1(s) test_sink_init((s), SINK_N_SHARDS, SHARD_CAP)

static int
test_sink_shard_count(const struct test_shard_sink* s)
{
  int count = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
    if (s->writers[0][i].buf && s->writers[0][i].size > 0)
      count++;
  return count;
}

// Helper: 2D config with given dtype. Shape 4x4, chunk 2x2, 2 shards along
// dim0, 1 along dim1 (cps 1x2). epoch_elements = 8.
static struct tile_stream_configuration
make_2d_config(struct dimension dims[2], enum dtype dt)
{
  dims_create(dims, "xy", (uint64_t[]){ 4, 4 });
  dims_set_chunk_sizes(dims, 2, (uint64_t[]){ 2, 2 });
  dims_set_shard_counts(dims, 2, (uint64_t[]){ 2, 1 });
  return (struct tile_stream_configuration){
    .buffer_capacity_bytes = 4096,
    .dtype = dt,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };
}

// Helper: write fill_byte data for exactly n elements of the given bpe.
static struct multiarray_writer_result
write_fill(struct multiarray_writer* w,
           int array_index,
           size_t n_elements,
           size_t bpe,
           uint8_t fill)
{
  size_t bytes = n_elements * bpe;
  uint8_t* data = (uint8_t*)malloc(bytes);
  if (!data)
    return (struct multiarray_writer_result){ .error = multiarray_writer_fail };
  memset(data, fill, bytes);
  struct slice sl = { .beg = data, .end = data + bytes };
  struct multiarray_writer_result r = w->update(w, array_index, sl);
  free(data);
  return r;
}

// ---- Test: basic two-array interleave ----

static int
test_basic_two_array(void)
{
  log_info("=== test_basic_two_array ===");

  struct test_shard_sink sink0, sink1;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);

  // Array 0: 3D 4x4x6, chunk 2x2x3, cps 1x2x2 → epoch_elements=48
  struct dimension dims0[3];
  dims_create(dims0, "zyx", (uint64_t[]){ 4, 4, 6 });
  dims_set_chunk_sizes(dims0, 3, (uint64_t[]){ 2, 2, 3 });
  dims_set_shard_counts(dims0, 3, (uint64_t[]){ 2, 1, 1 });
  struct tile_stream_configuration config0 = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims0,
    .codec = { .id = CODEC_NONE },
  };

  // Array 1: 2D 8x8, chunk 4x4, cps 1x2 → epoch_elements=32
  struct dimension dims1[2];
  dims_create(dims1, "xy", (uint64_t[]){ 8, 8 });
  dims_set_chunk_sizes(dims1, 2, (uint64_t[]){ 4, 4 });
  dims_set_shard_counts(dims1, 2, (uint64_t[]){ 2, 1 });
  struct tile_stream_configuration config1 = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims1,
    .codec = { .id = CODEC_NONE },
  };

  struct tile_stream_configuration configs[] = { config0, config1 };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);
  CHECK(Fail, w);

  // Write 1 epoch to array 0 (48 u16 elements).
  CHECK(Fail,
        write_fill(w, 0, 48, sizeof(uint16_t), 0xAA).error ==
          multiarray_writer_ok);

  // Write 1 epoch to array 1 (32 u16 elements).
  CHECK(Fail,
        write_fill(w, 1, 32, sizeof(uint16_t), 0xBB).error ==
          multiarray_writer_ok);

  struct multiarray_writer_result fr = w->flush(w);
  CHECK(Fail, fr.error == multiarray_writer_ok);

  CHECK(Fail, test_sink_shard_count(&sink0) > 0);
  CHECK(Fail, test_sink_shard_count(&sink1) > 0);
  log_info("  sink0: %d shards, sink1: %d shards",
           test_sink_shard_count(&sink0),
           test_sink_shard_count(&sink1));

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: switch at epoch boundary succeeds ----

static int
test_switch_at_epoch_boundary(void)
{
  log_info("=== test_switch_at_epoch_boundary ===");

  struct test_shard_sink sink0, sink1;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);

  struct dimension dims0[2], dims1[2];
  struct tile_stream_configuration config0 = make_2d_config(dims0, dtype_u8);
  struct tile_stream_configuration config1 = make_2d_config(dims1, dtype_u8);
  struct tile_stream_configuration configs[] = { config0, config1 };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write exactly 1 epoch (8 elements) to array 0.
  CHECK(Fail, write_fill(w, 0, 8, 1, 0xAB).error == multiarray_writer_ok);

  // Switch to array 1 should succeed (at epoch boundary).
  CHECK(Fail, write_fill(w, 1, 8, 1, 0xCD).error == multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: switch mid-epoch rejected ----

static int
test_switch_mid_epoch_rejected(void)
{
  log_info("=== test_switch_mid_epoch_rejected ===");

  struct test_shard_sink sink0, sink1;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);

  struct dimension dims0[2], dims1[2];
  struct tile_stream_configuration config0 = make_2d_config(dims0, dtype_u8);
  struct tile_stream_configuration config1 = make_2d_config(dims1, dtype_u8);
  struct tile_stream_configuration configs[] = { config0, config1 };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write half an epoch (4 of 8 elements) to array 0.
  CHECK(Fail, write_fill(w, 0, 4, 1, 0xCD).error == multiarray_writer_ok);

  // Try to switch to array 1 — should be rejected.
  {
    uint8_t buf[4] = { 0 };
    struct slice sl = { .beg = buf, .end = buf + 4 };
    struct multiarray_writer_result r = w->update(w, 1, sl);
    CHECK(Fail, r.error == multiarray_writer_not_flushable);
    CHECK(Fail, r.rest.beg == sl.beg);
  }

  // Finish the epoch and flush.
  CHECK(Fail, write_fill(w, 0, 4, 1, 0xEF).error == multiarray_writer_ok);
  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: flush all arrays ----

static int
test_flush_all(void)
{
  log_info("=== test_flush_all ===");

  struct test_shard_sink sink0, sink1, sink2;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);
  test_sink_init_1(&sink2);

  struct dimension d0[2], d1[2], d2[2];
  struct tile_stream_configuration configs[] = {
    make_2d_config(d0, dtype_u16),
    make_2d_config(d1, dtype_u16),
    make_2d_config(d2, dtype_u16),
  };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base, &sink2.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(3, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write 1 epoch (8 u16) to each array.
  for (int a = 0; a < 3; ++a)
    CHECK(Fail,
          write_fill(w, a, 8, sizeof(uint16_t), (uint8_t)(a + 1)).error ==
            multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  CHECK(Fail, test_sink_shard_count(&sink0) > 0);
  CHECK(Fail, test_sink_shard_count(&sink1) > 0);
  CHECK(Fail, test_sink_shard_count(&sink2) > 0);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  test_sink_free(&sink2);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  test_sink_free(&sink2);
  log_error("  FAIL");
  return 1;
}

// ---- Test: many arrays ----

static int
test_many_arrays(void)
{
  log_info("=== test_many_arrays ===");

  enum
  {
    N = 100
  };

  // All arrays share the same shape: 2D 4x4 u8.
  struct dimension dims[N][2];
  struct tile_stream_configuration* configs =
    (struct tile_stream_configuration*)malloc(
      N * sizeof(struct tile_stream_configuration));
  struct shard_sink** sinks_arr =
    (struct shard_sink**)malloc(N * sizeof(struct shard_sink*));
  struct test_shard_sink* mem_sinks =
    (struct test_shard_sink*)calloc(N, sizeof(struct test_shard_sink));
  struct multiarray_tile_stream_cpu* ms = NULL;
  CHECK(Fail, configs && sinks_arr && mem_sinks);

  for (int i = 0; i < N; ++i) {
    configs[i] = make_2d_config(dims[i], dtype_u8);
    test_sink_init_1(&mem_sinks[i]);
    sinks_arr[i] = &mem_sinks[i].base;
  }

  ms = multiarray_tile_stream_cpu_create(N, configs, sinks_arr, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = 8
  for (int i = 0; i < N; ++i)
    CHECK(Fail, write_fill(w, i, 8, 1, 0x42).error == multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  for (int i = 0; i < N; ++i)
    CHECK(Fail, test_sink_shard_count(&mem_sinks[i]) > 0);

  multiarray_tile_stream_cpu_destroy(ms);
  for (int i = 0; i < N; ++i)
    test_sink_free(&mem_sinks[i]);
  free(configs);
  free(sinks_arr);
  free(mem_sinks);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  if (mem_sinks)
    for (int i = 0; i < N; ++i)
      test_sink_free(&mem_sinks[i]);
  free(configs);
  free(sinks_arr);
  free(mem_sinks);
  log_error("  FAIL");
  return 1;
}

// ---- Test: same array repeated ----

static int
test_same_array_repeated(void)
{
  log_info("=== test_same_array_repeated ===");

  struct test_shard_sink sink;
  test_sink_init_1(&sink);

  // 2D 8x4, chunk 2x2, cps 1x2 → epoch_elements = 8, 4 epochs.
  struct dimension dims[2];
  dims_create(dims, "xy", (uint64_t[]){ 8, 4 });
  dims_set_chunk_sizes(dims, 2, (uint64_t[]){ 2, 2 });
  dims_set_shard_counts(dims, 2, (uint64_t[]){ 4, 1 });
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
  };

  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(1, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write 4 epochs of 8 u16.
  for (int epoch = 0; epoch < 4; ++epoch)
    CHECK(Fail,
          write_fill(w, 0, 8, sizeof(uint16_t), (uint8_t)epoch).error ==
            multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);
  CHECK(Fail, test_sink_shard_count(&sink) > 0);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// ---- Test: content isolation (cross-contamination check) ----

static int
test_content_isolation(void)
{
  log_info("=== test_content_isolation ===");

  struct test_shard_sink sink0, sink1;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);

  // Both arrays: 2D 4x4 u16, chunk 2x2, cps 1x2.
  // 2 epochs, 2 chunks/epoch, 2 chunks/shard → 2 shards, 4 chunks each.
  struct dimension dims0[2], dims1[2];
  struct tile_stream_configuration configs[] = {
    make_2d_config(dims0, dtype_u16),
    make_2d_config(dims1, dtype_u16),
  };
  struct shard_sink* sinks_arr[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks_arr, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Array 0: values 0x1000..0x100F (16 elements = 4x4).
  {
    uint16_t data[16];
    for (int i = 0; i < 16; ++i)
      data[i] = (uint16_t)(0x1000 + i);
    struct slice sl = { .beg = data, .end = (const char*)data + sizeof(data) };
    CHECK(Fail, w->update(w, 0, sl).error == multiarray_writer_ok);
  }

  // Array 1: values 0x2000..0x200F (16 elements = 4x4).
  {
    uint16_t data[16];
    for (int i = 0; i < 16; ++i)
      data[i] = (uint16_t)(0x2000 + i);
    struct slice sl = { .beg = data, .end = (const char*)data + sizeof(data) };
    CHECK(Fail, w->update(w, 1, sl).error == multiarray_writer_ok);
  }

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);
  CHECK(Fail, test_sink_shard_count(&sink0) > 0);
  CHECK(Fail, test_sink_shard_count(&sink1) > 0);

  // Shard layout: cps_total=2, index_tail=36 bytes, chunk_bytes=8
  const size_t cps_total = 2;
  const size_t index_tail = cps_total * 2 * sizeof(uint64_t) + 4;
  const size_t chunk_bytes = 4 * sizeof(uint16_t);

  // Verify array 0: all values in [0x1000, 0x100F].
  int chunks_0 = 0;
  for (int si = 0; si < TEST_SHARD_SINK_MAX_SHARDS; ++si) {
    struct test_shard_writer* sw = &sink0.writers[0][si];
    if (!sw->buf || sw->size == 0)
      continue;
    CHECK(Fail, sw->size >= index_tail);
    const uint64_t* idx = (const uint64_t*)(sw->buf + sw->size - index_tail);
    for (size_t c = 0; c < cps_total; ++c) {
      uint64_t off = idx[2 * c];
      uint64_t nb = idx[2 * c + 1];
      if (off == UINT64_MAX && nb == UINT64_MAX)
        continue;
      CHECK(Fail, nb == chunk_bytes);
      CHECK(Fail, off + nb <= sw->size - index_tail);
      const uint16_t* vals = (const uint16_t*)(sw->buf + off);
      for (size_t v = 0; v < nb / sizeof(uint16_t); ++v)
        CHECK(Fail, vals[v] >= 0x1000 && vals[v] <= 0x100F);
      chunks_0++;
    }
  }
  CHECK(Fail, chunks_0 == 4);

  // Verify array 1: all values in [0x2000, 0x200F].
  int chunks_1 = 0;
  for (int si = 0; si < TEST_SHARD_SINK_MAX_SHARDS; ++si) {
    struct test_shard_writer* sw = &sink1.writers[0][si];
    if (!sw->buf || sw->size == 0)
      continue;
    CHECK(Fail, sw->size >= index_tail);
    const uint64_t* idx = (const uint64_t*)(sw->buf + sw->size - index_tail);
    for (size_t c = 0; c < cps_total; ++c) {
      uint64_t off = idx[2 * c];
      uint64_t nb = idx[2 * c + 1];
      if (off == UINT64_MAX && nb == UINT64_MAX)
        continue;
      CHECK(Fail, nb == chunk_bytes);
      CHECK(Fail, off + nb <= sw->size - index_tail);
      const uint16_t* vals = (const uint16_t*)(sw->buf + off);
      for (size_t v = 0; v < nb / sizeof(uint16_t); ++v)
        CHECK(Fail, vals[v] >= 0x2000 && vals[v] <= 0x200F);
      chunks_1++;
    }
  }
  CHECK(Fail, chunks_1 == 4);

  log_info("  array0: %d chunks, array1: %d chunks", chunks_0, chunks_1);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: cross-validate multiarray (1 array) vs single-array ----

static int
test_cross_validate_single_array(void)
{
  log_info("=== test_cross_validate_single_array ===");

  struct test_shard_sink sink_ref, sink_multi;
  test_sink_init_1(&sink_ref);
  test_sink_init_1(&sink_multi);

  // 3D 4x4x6, chunk 2x2x3, cps 1x2x2, u16
  struct dimension dims_r[3], dims_m[3];
  for (int i = 0; i < 2; ++i) {
    struct dimension* d = i == 0 ? dims_r : dims_m;
    dims_create(d, "zyx", (uint64_t[]){ 4, 4, 6 });
    dims_set_chunk_sizes(d, 3, (uint64_t[]){ 2, 2, 3 });
    dims_set_shard_counts(d, 3, (uint64_t[]){ 2, 1, 1 });
  }
  struct tile_stream_configuration config_r = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims_r,
    .codec = { .id = CODEC_NONE },
  };
  struct tile_stream_configuration config_m = config_r;
  config_m.dimensions = dims_m;

  struct tile_stream_cpu* ref = NULL;
  struct multiarray_tile_stream_cpu* ms = NULL;

  // epoch_elements=48, 2 epochs = 96 total.
  size_t total_elems = 96;
  uint16_t* data = (uint16_t*)malloc(total_elems * sizeof(uint16_t));
  CHECK(Fail, data);
  for (size_t i = 0; i < total_elems; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  struct slice sl = {
    .beg = data,
    .end = (const char*)data + total_elems * sizeof(uint16_t),
  };

  // Single-array reference.
  ref = tile_stream_cpu_create(&config_r, &sink_ref.base);
  CHECK(Fail, ref);
  {
    struct writer* w = tile_stream_cpu_writer(ref);
    CHECK(Fail, w->append(w, sl).error == 0);
    CHECK(Fail, w->flush(w).error == 0);
  }

  // Multiarray (1 array).
  {
    struct tile_stream_configuration configs[] = { config_m };
    struct shard_sink* sinks[] = { &sink_multi.base };
    ms = multiarray_tile_stream_cpu_create(1, configs, sinks, 0);
    CHECK(Fail, ms);
    struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);
    CHECK(Fail, w->update(w, 0, sl).error == multiarray_writer_ok);
    CHECK(Fail, w->flush(w).error == multiarray_writer_ok);
  }

  // Compare all shards byte-for-byte.
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i) {
    struct test_shard_writer* sw_ref = &sink_ref.writers[0][i];
    struct test_shard_writer* sw_multi = &sink_multi.writers[0][i];
    int ref_has = sw_ref->buf && sw_ref->size > 0;
    int multi_has = sw_multi->buf && sw_multi->size > 0;
    CHECK(Fail, ref_has == multi_has);
    if (!ref_has)
      continue;
    CHECK(Fail, sw_ref->size == sw_multi->size);
    CHECK(Fail, memcmp(sw_ref->buf, sw_multi->buf, sw_ref->size) == 0);
  }

  free(data);
  tile_stream_cpu_destroy(ref);
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink_ref);
  test_sink_free(&sink_multi);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  tile_stream_cpu_destroy(ref);
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink_ref);
  test_sink_free(&sink_multi);
  log_error("  FAIL");
  return 1;
}

// ---- Test: minimal LOD (multiscale) ----

static int
test_lod_basic(void)
{
  log_info("=== test_lod_basic ===");

  // 3D shape 4x8x8, chunk 2x4x4, downsample dims 1,2.
  struct dimension dims[3];
  dims_create(dims, "zyx", (uint64_t[]){ 4, 8, 8 });
  dims_set_chunk_sizes(dims, 3, (uint64_t[]){ 2, 4, 4 });
  dims_set_shard_counts(dims, 3, (uint64_t[]){ 2, 1, 1 });
  dims_set_downsample_by_name(dims, 3, "yx");
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = { .id = CODEC_NONE },
    .reduce_method = lod_reduce_mean,
    .append_reduce_method = lod_reduce_mean,
    .epochs_per_batch = 1,
  };

  int shards_per_level[] = { 16, 16, 16 };
  struct test_shard_sink sink;
  test_sink_init_multi(&sink, 3, shards_per_level, SHARD_CAP);

  struct multiarray_tile_stream_cpu* ms = NULL;

  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };
  ms = multiarray_tile_stream_cpu_create(1, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = (2*4*4) * (2*2) = 128, 2 epochs = 256.
  CHECK(Fail,
        write_fill(w, 0, 256, sizeof(uint16_t), 0x11).error ==
          multiarray_writer_ok);
  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  int l0_count = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
    if (sink.writers[0][i].buf && sink.writers[0][i].size > 0)
      l0_count++;
  CHECK(Fail, l0_count > 0);

  int lod_count = 0;
  for (int lv = 1; lv < 3; ++lv)
    for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
      if (sink.writers[lv][i].buf && sink.writers[lv][i].size > 0)
        lod_count++;
  CHECK(Fail, lod_count > 0);

  log_info("  L0 shards: %d, L1+ shards: %d", l0_count, lod_count);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// ---- Test: mixed dtypes (u8 + u16) ----

static int
test_mixed_dtypes(void)
{
  log_info("=== test_mixed_dtypes ===");

  struct test_shard_sink sink0, sink1;
  test_sink_init_1(&sink0);
  test_sink_init_1(&sink1);

  struct dimension dims0[2], dims1[2];
  struct tile_stream_configuration configs[] = {
    make_2d_config(dims0, dtype_u8),
    make_2d_config(dims1, dtype_u16),
  };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = 8 for both, but bpe differs.
  CHECK(Fail,
        write_fill(w, 0, 8, sizeof(uint8_t), 0xAA).error ==
          multiarray_writer_ok);
  CHECK(Fail,
        write_fill(w, 1, 8, sizeof(uint16_t), 0xBB).error ==
          multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  CHECK(Fail, test_sink_shard_count(&sink0) > 0);
  CHECK(Fail, test_sink_shard_count(&sink1) > 0);
  log_info("  sink0 (u8): %d, sink1 (u16): %d",
           test_sink_shard_count(&sink0),
           test_sink_shard_count(&sink1));

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: mixed LOD (one multiscale, one not) ----

static int
test_mixed_lod(void)
{
  log_info("=== test_mixed_lod ===");

  // Array 0: plain 2D 4x4 u16 (no LOD).
  struct dimension dims0[2];
  struct tile_stream_configuration config0 = make_2d_config(dims0, dtype_u16);

  // Array 1: 3D 4x8x8, chunk 2x4x4, multiscale on y,x.
  struct dimension dims1[3];
  dims_create(dims1, "zyx", (uint64_t[]){ 4, 8, 8 });
  dims_set_chunk_sizes(dims1, 3, (uint64_t[]){ 2, 4, 4 });
  dims_set_shard_counts(dims1, 3, (uint64_t[]){ 2, 1, 1 });
  dims_set_downsample_by_name(dims1, 3, "yx");
  struct tile_stream_configuration config1 = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims1,
    .codec = { .id = CODEC_NONE },
    .reduce_method = lod_reduce_mean,
    .append_reduce_method = lod_reduce_mean,
    .epochs_per_batch = 1,
  };

  struct test_shard_sink sink0;
  test_sink_init_1(&sink0);

  int shards_per_level[] = { 16, 16, 16 };
  struct test_shard_sink sink1;
  test_sink_init_multi(&sink1, 3, shards_per_level, SHARD_CAP);

  struct tile_stream_configuration configs[] = { config0, config1 };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Array 0: 16 u16 elements (2 epochs * 8 elements).
  CHECK(Fail,
        write_fill(w, 0, 16, sizeof(uint16_t), 0x11).error ==
          multiarray_writer_ok);

  // Array 1: 256 u16 elements (2 epochs * 128 elements).
  CHECK(Fail,
        write_fill(w, 1, 256, sizeof(uint16_t), 0x22).error ==
          multiarray_writer_ok);

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  // Plain array produces L0 shards only.
  CHECK(Fail, test_sink_shard_count(&sink0) > 0);

  // Multiscale array produces L0 + L1+ shards.
  int l0 = 0, lod = 0;
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
    if (sink1.writers[0][i].buf && sink1.writers[0][i].size > 0)
      l0++;
  for (int lv = 1; lv < 3; ++lv)
    for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
      if (sink1.writers[lv][i].buf && sink1.writers[lv][i].size > 0)
        lod++;
  CHECK(Fail, l0 > 0);
  CHECK(Fail, lod > 0);
  log_info("  plain: %d shards, lod L0: %d L1+: %d",
           test_sink_shard_count(&sink0),
           l0,
           lod);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink0);
  test_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: write past max_cursor returns finished ----

static int
test_write_past_max_cursor(void)
{
  log_info("=== test_write_past_max_cursor ===");

  struct test_shard_sink sink;
  test_sink_init_1(&sink);

  // 2D 4x4 u8: epoch_elements=8, 2 epochs, max_cursor=16.
  struct dimension dims[2];
  struct tile_stream_configuration config = make_2d_config(dims, dtype_u8);

  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(1, configs, sinks, 0);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write exactly max_cursor (16 elements).
  CHECK(Fail, write_fill(w, 0, 16, 1, 0xAA).error == multiarray_writer_ok);

  // Writing more should return finished.
  {
    uint8_t extra[8] = { 0 };
    struct slice sl = { .beg = extra, .end = extra + sizeof(extra) };
    struct multiarray_writer_result r = w->update(w, 0, sl);
    CHECK(Fail, r.error == multiarray_writer_finished);
    CHECK(Fail, r.rest.beg == sl.beg); // all data unconsumed
  }

  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);
  CHECK(Fail, test_sink_shard_count(&sink) > 0);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// ---- Test: flush with no data written ----

static int
test_flush_no_data(void)
{
  log_info("=== test_flush_no_data ===");

  struct test_shard_sink sink;
  test_sink_init_1(&sink);

  struct dimension dims[2];
  struct tile_stream_configuration config = make_2d_config(dims, dtype_u8);
  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(1, configs, sinks, 0);
  CHECK(Fail, ms);

  // Flush immediately — no data written.
  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);
  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// ---- Test: metrics enabled ----

static int
test_metrics_enabled(void)
{
  log_info("=== test_metrics_enabled ===");

  struct test_shard_sink sink;
  test_sink_init_1(&sink);

  struct dimension dims[2];
  struct tile_stream_configuration config = make_2d_config(dims, dtype_u16);
  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(1, configs, sinks, 1);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write 2 epochs (16 u16 elements).
  CHECK(Fail,
        write_fill(w, 0, 16, sizeof(uint16_t), 0x42).error ==
          multiarray_writer_ok);
  CHECK(Fail, w->flush(w).error == multiarray_writer_ok);

  struct stream_metrics m = multiarray_tile_stream_cpu_get_metrics(ms);
  CHECK(Fail, m.compress.count > 0);
  CHECK(Fail, m.aggregate.count > 0);
  log_info("  compress: %lu calls, aggregate: %lu calls",
           (unsigned long)m.compress.count,
           (unsigned long)m.aggregate.count);

  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  test_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int rc = 0;
  rc |= test_basic_two_array();
  rc |= test_switch_at_epoch_boundary();
  rc |= test_switch_mid_epoch_rejected();
  rc |= test_flush_all();
  rc |= test_many_arrays();
  rc |= test_same_array_repeated();
  rc |= test_content_isolation();
  rc |= test_cross_validate_single_array();
  rc |= test_lod_basic();
  rc |= test_mixed_dtypes();
  rc |= test_mixed_lod();
  rc |= test_write_past_max_cursor();
  rc |= test_flush_no_data();
  rc |= test_metrics_enabled();
  return rc;
}
