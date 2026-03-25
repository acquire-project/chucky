#include "multiarray.cpu.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

// ---- Minimal in-memory shard sink ----

#define MAX_SHARDS 16
#define SHARD_CAP  (1 << 20)

struct mem_shard_writer
{
  struct shard_writer base;
  uint8_t* buf;
  size_t size;
  int finalized;
};

struct mem_shard_sink
{
  struct shard_sink base;
  struct mem_shard_writer writers[MAX_SHARDS];
  int finalize_count;
};

static int
mem_write(struct shard_writer* self,
          uint64_t offset,
          const void* beg,
          const void* end)
{
  struct mem_shard_writer* w = (struct mem_shard_writer*)self;
  size_t n = (size_t)((const char*)end - (const char*)beg);
  if (offset + n > SHARD_CAP)
    return 1;
  memcpy(w->buf + offset, beg, n);
  if (offset + n > w->size)
    w->size = offset + n;
  return 0;
}

static int
mem_finalize(struct shard_writer* self)
{
  struct mem_shard_writer* w = (struct mem_shard_writer*)self;
  w->finalized = 1;
  return 0;
}

static struct shard_writer*
mem_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct mem_shard_sink* s = (struct mem_shard_sink*)self;
  if (shard_index >= MAX_SHARDS)
    return NULL;
  struct mem_shard_writer* w = &s->writers[shard_index];
  if (!w->buf) {
    w->buf = (uint8_t*)calloc(1, SHARD_CAP);
    if (!w->buf)
      return NULL;
    w->base.write = mem_write;
    w->base.finalize = mem_finalize;
  }
  w->finalized = 0;
  w->size = 0;
  return &w->base;
}

static void
mem_sink_init(struct mem_shard_sink* s)
{
  memset(s, 0, sizeof(*s));
  s->base.open = mem_open;
}

static void
mem_sink_free(struct mem_shard_sink* s)
{
  for (int i = 0; i < MAX_SHARDS; ++i)
    free(s->writers[i].buf);
}

static int
mem_sink_shard_count(const struct mem_shard_sink* s)
{
  int count = 0;
  for (int i = 0; i < MAX_SHARDS; ++i)
    if (s->writers[i].buf && s->writers[i].size > 0)
      count++;
  return count;
}

// ---- Test: basic two-array interleave ----

static int
test_basic_two_array(void)
{
  log_info("=== test_multiarray_basic_two_array ===");

  struct mem_shard_sink sink0, sink1;
  mem_sink_init(&sink0);
  mem_sink_init(&sink1);

  // Array 0: 3D 4x4x6, chunk 2x2x3, cps 1x2x2
  struct dimension dims0[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
    { .size = 6, .chunk_size = 3, .chunks_per_shard = 2,
      .storage_position = 2 },
  };
  struct tile_stream_configuration config0 = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims0,
    .codec = CODEC_NONE,
  };

  // Array 1: 2D 8x8, chunk 4x4, cps 1x2
  struct dimension dims1[] = {
    { .size = 8, .chunk_size = 4, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 8, .chunk_size = 4, .chunks_per_shard = 2,
      .storage_position = 1 },
  };
  struct tile_stream_configuration config1 = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims1,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration configs[] = { config0, config1 };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);
  CHECK(Fail, w);

  // Write 1 epoch to array 0.
  // epoch_elements for config0: 2*2*3 * (4/2)*(6/3) = 12 * 4 = 48 (per epoch-chunk * chunks)
  // Actually, epoch_elements = chunk_elements * chunks_per_epoch
  // For this config with K likely 1, just write enough for 1 epoch.
  // Compute manually: epoch_elements = prod(chunk_size) * prod(ceil(size[d]/cs[d]) for d>0)
  // = (2*2*3) * (2*2) = 12*4 = 48. For dims[0].chunk_size=2, epoch = 48 elements.
  {
    size_t n = 48;
    uint16_t* data = (uint16_t*)malloc(n * sizeof(uint16_t));
    CHECK(Fail, data);
    for (size_t i = 0; i < n; ++i)
      data[i] = (uint16_t)(i & 0xFFFF);

    struct slice sl = { .beg = data,
                        .end = (const char*)data + n * sizeof(uint16_t) };
    struct multiarray_writer_result r = w->update(w, 0, sl);
    CHECK(Fail, r.error == multiarray_writer_ok);
    free(data);
  }

  // Write 1 epoch to array 1.
  // epoch_elements = (4*4) * (8/4) = 16 * 2 = 32
  {
    size_t n = 32;
    uint16_t* data = (uint16_t*)malloc(n * sizeof(uint16_t));
    CHECK(Fail, data);
    for (size_t i = 0; i < n; ++i)
      data[i] = (uint16_t)((i + 100) & 0xFFFF);

    struct slice sl = { .beg = data,
                        .end = (const char*)data + n * sizeof(uint16_t) };
    struct multiarray_writer_result r = w->update(w, 1, sl);
    CHECK(Fail, r.error == multiarray_writer_ok);
    free(data);
  }

  // Flush all.
  struct multiarray_writer_result fr = w->flush(w);
  CHECK(Fail, fr.error == multiarray_writer_ok);

  // Verify shards written for both arrays.
  CHECK(Fail, mem_sink_shard_count(&sink0) > 0);
  CHECK(Fail, mem_sink_shard_count(&sink1) > 0);
  log_info("  sink0: %d shards, sink1: %d shards",
           mem_sink_shard_count(&sink0),
           mem_sink_shard_count(&sink1));

  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: switch at epoch boundary succeeds ----

static int
test_switch_at_epoch_boundary(void)
{
  log_info("=== test_switch_at_epoch_boundary ===");

  struct mem_shard_sink sink0, sink1;
  mem_sink_init(&sink0);
  mem_sink_init(&sink1);

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u8,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration configs[] = { config, config };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  uint8_t* data = NULL;
  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = chunk_elements * chunks_per_epoch = (2*2) * (4/2) = 4 * 2 = 8
  size_t n = 8;
  data = (uint8_t*)malloc(n);
  CHECK(Fail, data);
  memset(data, 0xAB, n);

  // Write exactly 1 epoch to array 0.
  struct slice sl = { .beg = data, .end = data + n };
  struct multiarray_writer_result r = w->update(w, 0, sl);
  CHECK(Fail, r.error == multiarray_writer_ok);

  // Switch to array 1 should succeed (at epoch boundary).
  r = w->update(w, 1, sl);
  CHECK(Fail, r.error == multiarray_writer_ok);

  r = w->flush(w);
  CHECK(Fail, r.error == multiarray_writer_ok);

  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: switch mid-epoch rejected ----

static int
test_switch_mid_epoch_rejected(void)
{
  log_info("=== test_switch_mid_epoch_rejected ===");

  struct mem_shard_sink sink0, sink1;
  mem_sink_init(&sink0);
  mem_sink_init(&sink1);

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u8,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration configs[] = { config, config };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base };

  uint8_t* data = NULL;
  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(2, configs, sinks);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // Write half an epoch to array 0 (epoch_elements = 8, write 4).
  size_t n = 4;
  data = (uint8_t*)malloc(n);
  CHECK(Fail, data);
  memset(data, 0xCD, n);

  struct slice sl = { .beg = data, .end = data + n };
  struct multiarray_writer_result r = w->update(w, 0, sl);
  CHECK(Fail, r.error == multiarray_writer_ok);

  // Try to switch to array 1 — should be rejected.
  r = w->update(w, 1, sl);
  CHECK(Fail, r.error == multiarray_writer_not_flushable);

  // The rest should be the entire input (no bytes consumed).
  CHECK(Fail, r.rest.beg == sl.beg);

  // Clean up: finish the epoch and flush.
  r = w->update(w, 0, sl);
  CHECK(Fail, r.error == multiarray_writer_ok);

  r = w->flush(w);
  CHECK(Fail, r.error == multiarray_writer_ok);

  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  log_error("  FAIL");
  return 1;
}

// ---- Test: flush all arrays ----

static int
test_flush_all(void)
{
  log_info("=== test_flush_all ===");

  struct mem_shard_sink sink0, sink1, sink2;
  mem_sink_init(&sink0);
  mem_sink_init(&sink1);
  mem_sink_init(&sink2);

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration configs[] = { config, config, config };
  struct shard_sink* sinks[] = { &sink0.base, &sink1.base, &sink2.base };

  uint16_t* data = NULL;
  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(3, configs, sinks);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = (2*2) * (4/2) = 4 * 2 = 8
  size_t n = 8;
  data = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, data);
  for (size_t i = 0; i < n; ++i)
    data[i] = (uint16_t)i;

  struct slice sl = { .beg = data,
                      .end = (const char*)data + n * sizeof(uint16_t) };

  // Write 1 epoch to each array.
  for (int a = 0; a < 3; ++a) {
    struct multiarray_writer_result r = w->update(w, a, sl);
    CHECK(Fail, r.error == multiarray_writer_ok);
  }

  // Flush everything.
  struct multiarray_writer_result fr = w->flush(w);
  CHECK(Fail, fr.error == multiarray_writer_ok);

  CHECK(Fail, mem_sink_shard_count(&sink0) > 0);
  CHECK(Fail, mem_sink_shard_count(&sink1) > 0);
  CHECK(Fail, mem_sink_shard_count(&sink2) > 0);

  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  mem_sink_free(&sink2);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink0);
  mem_sink_free(&sink1);
  mem_sink_free(&sink2);
  log_error("  FAIL");
  return 1;
}

// ---- Test: many arrays ----

static int
test_many_arrays(void)
{
  log_info("=== test_many_arrays ===");

  enum { N = 100 };

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u8,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration* configs =
    (struct tile_stream_configuration*)malloc(
      N * sizeof(struct tile_stream_configuration));
  struct shard_sink** sinks_arr =
    (struct shard_sink**)malloc(N * sizeof(struct shard_sink*));
  struct mem_shard_sink* mem_sinks =
    (struct mem_shard_sink*)calloc(N, sizeof(struct mem_shard_sink));
  struct multiarray_tile_stream_cpu* ms = NULL;
  CHECK(Fail, configs && sinks_arr && mem_sinks);

  for (int i = 0; i < N; ++i) {
    configs[i] = config;
    mem_sink_init(&mem_sinks[i]);
    sinks_arr[i] = &mem_sinks[i].base;
  }

  ms = multiarray_tile_stream_cpu_create(N, configs, sinks_arr);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = 8
  size_t n = 8;
  uint8_t* data = (uint8_t*)malloc(n);
  CHECK(Fail, data);
  memset(data, 0x42, n);

  struct slice sl = { .beg = data, .end = data + n };
  for (int i = 0; i < N; ++i) {
    struct multiarray_writer_result r = w->update(w, i, sl);
    CHECK(Fail, r.error == multiarray_writer_ok);
  }

  struct multiarray_writer_result fr = w->flush(w);
  CHECK(Fail, fr.error == multiarray_writer_ok);

  // Verify all arrays produced shards.
  for (int i = 0; i < N; ++i)
    CHECK(Fail, mem_sink_shard_count(&mem_sinks[i]) > 0);

  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  for (int i = 0; i < N; ++i)
    mem_sink_free(&mem_sinks[i]);
  free(configs);
  free(sinks_arr);
  free(mem_sinks);
  log_info("  PASS");
  return 0;

Fail:
  multiarray_tile_stream_cpu_destroy(ms);
  if (mem_sinks)
    for (int i = 0; i < N; ++i)
      mem_sink_free(&mem_sinks[i]);
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

  struct mem_shard_sink sink;
  mem_sink_init(&sink);

  struct dimension dims[] = {
    { .size = 8, .chunk_size = 2, .chunks_per_shard = 1,
      .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2,
      .storage_position = 1 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_configuration configs[] = { config };
  struct shard_sink* sinks[] = { &sink.base };

  uint16_t* data = NULL;
  struct multiarray_tile_stream_cpu* ms =
    multiarray_tile_stream_cpu_create(1, configs, sinks);
  CHECK(Fail, ms);

  struct multiarray_writer* w = multiarray_tile_stream_cpu_writer(ms);

  // epoch_elements = (2*2) * (4/2) = 4*2 = 8
  // Write 4 epochs worth (total = 4*8 = 32 elements)
  size_t per_epoch = 8;
  size_t n = 4 * per_epoch;
  data = (uint16_t*)malloc(n * sizeof(uint16_t));
  CHECK(Fail, data);
  for (size_t i = 0; i < n; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  // Write in 4 separate calls to same array.
  for (int epoch = 0; epoch < 4; ++epoch) {
    uint16_t* p = data + epoch * per_epoch;
    struct slice sl = {
      .beg = p,
      .end = (const char*)p + per_epoch * sizeof(uint16_t),
    };
    struct multiarray_writer_result r = w->update(w, 0, sl);
    CHECK(Fail, r.error == multiarray_writer_ok);
  }

  struct multiarray_writer_result fr = w->flush(w);
  CHECK(Fail, fr.error == multiarray_writer_ok);

  CHECK(Fail, mem_sink_shard_count(&sink) > 0);

  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  free(data);
  multiarray_tile_stream_cpu_destroy(ms);
  mem_sink_free(&sink);
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
  return rc;
}
