#include "cpu/stream.h"
#include "crc32c.h"
#include "defs.limits.h"
#include "prelude.h"

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
mem_write(struct shard_writer* self, uint64_t offset, const void* beg, const void* end)
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

// ---- Test: write 2 epochs of u16 data, verify shards are non-empty ----

static int
test_basic_pipeline(void)
{
  log_info("=== test_stream_cpu_basic ===");

  struct mem_shard_sink sink;
  mem_sink_init(&sink);

  // 3D: 4×4×6, chunk 2×2×3, chunks_per_shard = 1×2×2
  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1, .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2, .storage_position = 1 },
    { .size = 6, .chunk_size = 3, .chunks_per_shard = 2, .storage_position = 2 },
  };

  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_u16,
    .rank = 3,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s);

  const struct tile_stream_layout* lay = tile_stream_cpu_layout(s);
  log_info("  epoch_elements=%lu chunks_per_epoch=%lu",
           (unsigned long)lay->epoch_elements,
           (unsigned long)lay->chunks_per_epoch);

  // Write 2 epochs of sequential u16 data.
  uint64_t total_elements = 2 * lay->epoch_elements;
  size_t total_bytes = total_elements * sizeof(uint16_t);
  uint16_t* data = (uint16_t*)malloc(total_bytes);
  CHECK(Fail, data);
  for (uint64_t i = 0; i < total_elements; ++i)
    data[i] = (uint16_t)(i & 0xFFFF);

  struct writer* w = tile_stream_cpu_writer(s);
  struct slice sl = { .beg = data, .end = (const char*)data + total_bytes };
  struct writer_result r = writer_append(w, sl);
  CHECK(Fail, r.error == 0);

  r = writer_flush(w);
  CHECK(Fail, r.error == 0);

  CHECK(Fail, tile_stream_cpu_cursor(s) == total_elements);

  // Verify at least one shard was written.
  int found = 0;
  for (int i = 0; i < MAX_SHARDS; ++i) {
    if (sink.writers[i].buf && sink.writers[i].size > 0) {
      found = 1;
      log_info("  shard %d: %lu bytes", i, (unsigned long)sink.writers[i].size);
    }
  }
  CHECK(Fail, found);

  // Metrics sanity.
  struct stream_metrics m = tile_stream_cpu_get_metrics(s);
  CHECK(Fail, m.scatter.count > 0);
  CHECK(Fail, m.compress.count > 0);
  CHECK(Fail, m.sink.count > 0);

  free(data);
  tile_stream_cpu_destroy(s);
  mem_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_cpu_destroy(s);
  mem_sink_free(&sink);
  log_error("  FAIL");
  return 1;
}

// Test f16 rejection.
static int
test_f16_rejected(void)
{
  log_info("=== test_stream_cpu_f16 ===");

  struct mem_shard_sink sink;
  mem_sink_init(&sink);

  struct dimension dims[] = {
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 1, .storage_position = 0 },
    { .size = 4, .chunk_size = 2, .chunks_per_shard = 2, .storage_position = 1 },
  };
  struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 4096,
    .dtype = dtype_f16,
    .rank = 2,
    .dimensions = dims,
    .codec = CODEC_NONE,
  };

  struct tile_stream_cpu* s = tile_stream_cpu_create(&config, &sink.base);
  CHECK(Fail, s == NULL); // should be rejected

  mem_sink_free(&sink);
  log_info("  PASS");
  return 0;

Fail:
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
  rc |= test_basic_pipeline();
  rc |= test_f16_rejected();
  return rc;
}
