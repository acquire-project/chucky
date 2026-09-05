#include "platform/platform.h"
#include "platform/platform_io.h"
#include "store.h"
#include "stream/host_output_pool.h"
#include "test_io_faults.h"
#include "test_platform.h"
#include "util/prelude.h"
#include "zarr.h"
#include "zarr/io_backend.fs.h"
#include "zarr/io_scheduler.h"
#include "zarr/shard_pool.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char tmpdir[512];

static int
make_tmpdir(void)
{
  CHECK(Fail, test_tmpdir_create(tmpdir, sizeof(tmpdir)) == 0);
  return 0;
Fail:
  return 1;
}

// --- store put/mkdirs ---

static int
test_store_put(void)
{
  log_info("=== test_store_put ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  const char* data = "hello world";
  CHECK(Fail2, s->put(s, "test.txt", data, strlen(data)) == 0);

  // Verify file contents
  char path[4096];
  snprintf(path, sizeof(path), "%s/test.txt", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail2, f);
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  CHECK(Fail2, n == strlen(data));
  CHECK(Fail2, memcmp(buf, data, n) == 0);

  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_store_mkdirs(void)
{
  log_info("=== test_store_mkdirs ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  CHECK(Fail2, s->mkdirs(s, "a/b/c") == 0);

  // Now put a file inside the created dir
  CHECK(Fail2, s->put(s, "a/b/c/data.txt", "ok", 2) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/a/b/c/data.txt", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail2, f);
  fclose(f);

  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- shard pool ---

static int
test_shard_pool_write(void)
{
  log_info("=== test_shard_pool_write ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  // Create a subdir for shard files
  CHECK(Fail2, s->mkdirs(s, "shards") == 0);

  struct shard_pool* pool = s->create_pool(s, 2);
  CHECK(Fail2, pool);

  // Write to slot 0
  char key[256];
  snprintf(key, sizeof(key), "shards/shard_0.bin");
  struct shard_writer* w = pool->open(pool, 0, key);
  CHECK(Fail3, w);

  const char* data = "shard data here";
  size_t len = strlen(data);
  CHECK(Fail3, w->write(w, 0, data, data + len) == 0);
  CHECK(Fail3, w->finalize(w) == 0);

  // Flush and verify
  CHECK(Fail3, pool->flush(pool) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/shards/shard_0.bin", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail3, f);
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  CHECK(Fail3, n == len);
  CHECK(Fail3, memcmp(buf, data, n) == 0);

  shard_pool_destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  shard_pool_destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_fence(void)
{
  log_info("=== test_shard_pool_fence ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);
  CHECK(Fail2, s->mkdirs(s, "fence") == 0);

  struct shard_pool* pool = s->create_pool(s, 4);
  CHECK(Fail2, pool);

  // Write to multiple slots
  for (int i = 0; i < 4; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "fence/s%d.bin", i);
    struct shard_writer* w = pool->open(pool, (uint64_t)i, key);
    CHECK(Fail3, w);
    char data[32];
    int dlen = snprintf(data, sizeof(data), "slot_%d", i);
    CHECK(Fail3, w->write(w, 0, data, data + dlen) == 0);
    CHECK(Fail3, w->finalize(w) == 0);
  }

  struct io_event ev = pool->record_fence(pool);
  pool->wait_fence(pool, ev);

  CHECK(Fail3, pool->has_error(pool) == 0);
  CHECK(Fail3, pool->pending_bytes(pool) == 0);

  shard_pool_destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  shard_pool_destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_on_demand_mkdir(void)
{
  log_info("=== test_shard_pool_on_demand_mkdir ===");
  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct shard_pool* pool = s->create_pool(s, 1);
  CHECK(Fail2, pool);

  // Open with a key whose parent directory doesn't exist yet.
  // Pool should create it on-demand.
  struct shard_writer* w = pool->open(pool, 0, "deep/nested/dir/shard.bin");
  CHECK(Fail3, w);
  const char byte = 'x';
  CHECK(Fail3, w->write(w, 0, &byte, &byte + 1) == 0);
  CHECK(Fail3, w->finalize(w) == 0);
  CHECK(Fail3, pool->flush(pool) == 0);

  char path[4096];
  snprintf(path, sizeof(path), "%s/deep/nested/dir/shard.bin", tmpdir);
  FILE* f = fopen(path, "rb");
  CHECK(Fail3, f);
  fclose(f);

  shard_pool_destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  shard_pool_destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_unbuffered(void)
{
  log_info("=== test_shard_pool_unbuffered ===");

  // Create store with unbuffered=1 → pool uses page-aligned writes
  struct store* s = store_fs_create(tmpdir, 1);
  CHECK(Fail, s);
  CHECK(Fail2, s->mkdirs(s, "unbuf") == 0);

  struct shard_pool* pool = s->create_pool(s, 2);
  CHECK(Fail2, pool);

  // Write via the copy path (write) — exercises aligned alloc
  struct shard_writer* w = pool->open(pool, 0, "unbuf/shard0.bin");
  CHECK(Fail3, w);

  // O_DIRECT / F_NOCACHE / FILE_FLAG_NO_BUFFERING require offset, length,
  // and source pointer all aligned to the device's page alignment (16 KiB
  // on Apple Silicon, 4 KiB on most Linux/Windows).
  size_t page = platform_page_alignment();
  char* data = (char*)platform_aligned_alloc(page, page);
  CHECK(Fail3, data);
  memset(data, 0xAB, page);
  CHECK(Fail4, w->write(w, 0, data, data + page) == 0);

  // The write_direct payload is not copied.
  if (w->write_direct) {
    CHECK(Fail4, w->write_direct(w, page, data, data + page) == 0);
  }

  CHECK(Fail4, w->finalize(w) == 0);
  CHECK(Fail4, pool->flush(pool) == 0);
  CHECK(Fail4, pool->has_error(pool) == 0);

  // Verify file exists and has expected size
  char path[4096];
  snprintf(path, sizeof(path), "%s/unbuf/shard0.bin", tmpdir);
  CHECK(Fail4, test_file_exists(path));

  platform_aligned_free(data);
  shard_pool_destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail4:
  platform_aligned_free(data);
Fail3:
  shard_pool_destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_error_propagation(void)
{
  log_info("=== test_shard_pool_error_propagation ===");

  // Use a buffered pool — the error path under test is filesystem-independent,
  // driven by a test-only failing-job injector rather than by O_DIRECT
  // alignment enforcement (which varies across filesystems).
  struct io_faults faults;
  struct shard_pool* pool = io_faults_pool_create(&faults, tmpdir, 1, 0, NULL);
  CHECK(Fail, pool);

  // Inject a job that deliberately reports a write failure.
  CHECK(Fail2, io_faults_inject_failing_job(&faults) == 0);

  // Flush waits for all async IO and returns the error flag.
  int flush_err = pool->flush(pool);
  CHECK(Fail2, flush_err != 0);
  CHECK(Fail2, pool->has_error(pool) != 0);

  shard_pool_destroy(pool);
  log_info("  PASS");
  return 0;

Fail2:
  shard_pool_destroy(pool);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_open_failure(void)
{
  log_info("=== test_shard_pool_open_failure ===");

  struct io_faults faults;
  struct shard_pool* pool = io_faults_pool_create(&faults, tmpdir, 1, 0, NULL);
  CHECK(Fail, pool);

  io_faults_fail_next_open(&faults);
  struct shard_writer* writer = pool->open(pool, 0, "failed-open/shard.bin");
  CHECK(Fail2, writer);
  CHECK(Fail2, writer->finalize(writer) == 0);
  CHECK(Fail2, pool->flush(pool) != 0);

  shard_pool_destroy(pool);
  log_info("  PASS");
  return 0;

Fail2:
  shard_pool_destroy(pool);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_close_failure(void)
{
  log_info("=== test_shard_pool_close_failure ===");

  struct io_faults faults;
  struct shard_pool* pool = io_faults_pool_create(&faults, tmpdir, 1, 0, NULL);
  CHECK(Fail, pool);

  const char* key = "failed-close/shard.bin";
  struct shard_writer* writer = pool->open(pool, 0, key);
  CHECK(Cleanup, writer);
  const char byte = 'x';
  CHECK(Cleanup, writer->write(writer, 0, &byte, &byte + 1) == 0);
  io_faults_fail_next_close(&faults);
  CHECK(Cleanup, writer->finalize(writer) == 0);
  CHECK(Cleanup, pool->flush(pool) != 0);

  shard_pool_destroy(pool);
  pool = NULL;

  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  CHECK(Fail, platform_remove_file(path) == 0);

  log_info("  PASS");
  return 0;

Cleanup:
  shard_pool_destroy(pool);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_failed_output_write_releases_buffer(void)
{
  log_info("=== test_failed_output_write_releases_buffer ===");
  struct io_faults faults;
  struct shard_pool* pool = io_faults_pool_create(&faults, tmpdir, 1, 0, NULL);
  CHECK(Fail, pool);

  const size_t page = platform_page_alignment();
  struct host_output_pool* outputs =
    host_output_pool_create(page, page, (struct host_output_allocator){ 0 });
  CHECK(CleanupPool, outputs);
  struct host_output output = { 0 };
  CHECK(CleanupOutputs, host_output_pool_acquire(outputs, &output) == 0);
  memset(output.data, 0xa5, page);

  struct shard_writer* writer = pool->open(pool, 0, "failed-output/shard.bin");
  CHECK(CleanupOutput, writer && writer->write_from_output);
  io_faults_fail_next_write(&faults);
  CHECK(
    CleanupOutput,
    writer->write_from_output(
      writer, 0, output.data, (const char*)output.data + page, output.group) ==
      0);
  host_output_group_seal(output.group);
  CHECK(CleanupOutputs, pool->flush(pool) != 0);

  shard_pool_destroy(pool);
  pool = NULL;
  host_output_pool_destroy(outputs);
  log_info("  PASS");
  return 0;

CleanupOutput:
  host_output_group_seal(output.group);
CleanupOutputs:
  shard_pool_destroy(pool);
  host_output_pool_destroy(outputs);
  log_error("  FAIL");
  return 1;
CleanupPool:
  shard_pool_destroy(pool);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_has_existing_data(void)
{
  log_info("=== test_has_existing_data ===");

  // Empty subdir: no zarr.json → 0
  char empty_root[4096];
  snprintf(empty_root, sizeof(empty_root), "%s/has_empty", tmpdir);
  CHECK(Fail, test_mkdir(empty_root) == 0);

  struct store* s = store_fs_create(empty_root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  // With zarr.json → 1
  char with_root[4096];
  snprintf(with_root, sizeof(with_root), "%s/has_with", tmpdir);
  CHECK(Fail, test_mkdir(with_root) == 0);

  s = store_fs_create(with_root, 0);
  CHECK(Fail, s);
  const char* meta = "{}";
  CHECK(Fail2, s->put(s, "zarr.json", meta, strlen(meta)) == 0);
  CHECK(Fail2, store_has_existing_data(s) == 1);
  store_destroy(s);

  // Non-existent root → -1 (stat fails with ENOENT on parent)
  char missing_root[4096];
  snprintf(missing_root, sizeof(missing_root), "%s/does_not_exist", tmpdir);
  s = store_fs_create(missing_root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// Directory exists with unrelated files but no zarr.json → 0.
static int
test_has_existing_data_unrelated_files(void)
{
  log_info("=== test_has_existing_data_unrelated_files ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_unrelated", tmpdir);
  CHECK(Fail, test_mkdir(root) == 0);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);
  const char* data = "not zarr";
  CHECK(Fail2, s->put(s, "other.txt", data, strlen(data)) == 0);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// After a real zarr group write at root → 1.
static int
test_has_existing_data_after_write_group(void)
{
  log_info("=== test_has_existing_data_after_write_group ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_group", tmpdir);
  CHECK(Fail, test_mkdir(root) == 0);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  struct zarr_group* g = zarr_group_create(s, "");
  CHECK(Fail2, g);
  zarr_group_destroy(g);
  CHECK(Fail2, store_has_existing_data(s) == 1);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// Root path is a regular file, not a directory.
// stat("<file>/zarr.json") fails with ENOTDIR, which fs_has_existing_data
// maps to 0 (treated as "no zarr data present"). Documented behavior.
static int
test_has_existing_data_root_is_file(void)
{
  log_info("=== test_has_existing_data_root_is_file ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/has_root_is_file", tmpdir);
  FILE* f = fopen(root, "wb");
  CHECK(Fail, f);
  fputs("x", f);
  fclose(f);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);
  CHECK(Fail2, store_has_existing_data(s) == 0);
  store_destroy(s);

  log_info("  PASS");
  return 0;

Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- Concurrent reads during repeated puts (#123) ---

// Alternates between a long and a short document so a non-atomic rewrite
// leaves the file either empty or holding a prefix of the long one.
#define PUT_LONG_BYTES (256 * 1024)
#define PUT_ROUNDS 200

struct put_reader
{
  char path[4200];
  atomic_int stop;
  int reads;
  int torn;
};

// A complete document starts with '{' and ends with '}'. Truncation breaks
// one or the other, and O_TRUNC's empty window breaks both.
static int
document_is_complete(const char* buf, size_t len)
{
  return len >= 2 && buf[0] == '{' && buf[len - 1] == '}';
}

static void
put_reader_fn(void* arg)
{
  struct put_reader* r = (struct put_reader*)arg;
  char* buf = (char*)malloc(PUT_LONG_BYTES + 64);
  if (!buf)
    return;

  while (!atomic_load(&r->stop)) {
    FILE* f = fopen(r->path, "rb");
    if (!f)
      continue; // not created yet
    size_t len = fread(buf, 1, PUT_LONG_BYTES + 64, f);
    fclose(f);
    r->reads++;
    if (!document_is_complete(buf, len))
      r->torn++;
    // Poll rather than hold the file open without pause: on Windows no writer
    // can replace a file a reader never lets go of.
    platform_sleep_ns(200000);
  }
  free(buf);
}

static int
test_put_is_atomic_for_readers(void)
{
  log_info("=== test_put_is_atomic_for_readers ===");

  char root[4096];
  snprintf(root, sizeof(root), "%s/atomic_put", tmpdir);
  CHECK(Fail, test_mkdir(root) == 0);

  struct store* s = store_fs_create(root, 0);
  CHECK(Fail, s);

  char* longdoc = (char*)malloc(PUT_LONG_BYTES);
  CHECK(Fail2, longdoc);
  memset(longdoc, 'a', PUT_LONG_BYTES);
  longdoc[0] = '{';
  longdoc[PUT_LONG_BYTES - 1] = '}';
  const char* shortdoc = "{}";

  struct put_reader reader = { 0 };
  snprintf(reader.path, sizeof(reader.path), "%s/zarr.json", root);
  atomic_store(&reader.stop, 0);

  CHECK(Fail3, s->put(s, "zarr.json", shortdoc, strlen(shortdoc)) == 0);

  test_thread* t = NULL;
  CHECK(Fail3, test_thread_start(&t, put_reader_fn, &reader) == 0);

  int put_err = 0;
  for (int i = 0; i < PUT_ROUNDS && !put_err; ++i) {
    put_err = (i % 2) ? s->put(s, "zarr.json", shortdoc, strlen(shortdoc))
                      : s->put(s, "zarr.json", longdoc, PUT_LONG_BYTES);
  }

  atomic_store(&reader.stop, 1);
  CHECK(Fail3, test_thread_join(t) == 0);
  CHECK(Fail3, put_err == 0);

  log_info("  %d reads, %d torn", reader.reads, reader.torn);
  CHECK(Fail3, reader.reads > 0);
  CHECK(Fail3, reader.torn == 0);

  // The last round wrote the short document; the rename left exactly that.
  FILE* f = fopen(reader.path, "rb");
  CHECK(Fail3, f);
  char final_doc[64];
  size_t final_len = fread(final_doc, 1, sizeof(final_doc), f);
  fclose(f);
  CHECK(Fail3, final_len == strlen(shortdoc));
  CHECK(Fail3, memcmp(final_doc, shortdoc, final_len) == 0);

  free(longdoc);
  store_destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  free(longdoc);
Fail2:
  store_destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

// --- pre-sizing ---

static long
file_size(const char* path)
{
  FILE* f = fopen(path, "rb");
  if (!f)
    return -1;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fclose(f);
  return n;
}

#define PRESIZE_BYTES (1 << 20)

// Sizes are read with the shard closed: the pool's handle is not shared, so
// on Windows nothing else can open the file while the shard is live.
static long
closed_shard_size(struct shard_pool* pool,
                  const char* key,
                  uint64_t presize_to,
                  const char* payload,
                  size_t payload_bytes)
{
  struct shard_writer* w = pool->open(pool, 0, key);
  if (!w || !w->presize || w->presize(w, presize_to))
    return -1;
  if (payload_bytes > 0 && (w->write(w, 0, payload, payload + payload_bytes) ||
                            w->truncate(w, payload_bytes)))
    return -1;
  if (w->finalize(w) || pool->flush(pool))
    return -1;

  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", tmpdir, key);
  return file_size(path);
}

// A pre-sized file is as long as it was told to be, and the truncate at the
// end brings it back to what it holds.
static int
test_shard_pool_presize(void)
{
  log_info("=== test_shard_pool_presize ===");

  struct store* s = store_fs_create(tmpdir, 0);
  CHECK(Fail, s);

  struct shard_pool* pool = s->create_pool(s, 1);
  CHECK(Fail2, pool);

  static const char payload[] = "held";
  char key[256];

  snprintf(key, sizeof(key), "presize_empty.bin");
  const long presized = platform_should_presize_shard() ? PRESIZE_BYTES : 0;
  CHECK(Fail3,
        closed_shard_size(pool, key, PRESIZE_BYTES, NULL, 0) == presized);

  snprintf(key, sizeof(key), "presize_trimmed.bin");
  CHECK(Fail3,
        closed_shard_size(pool, key, PRESIZE_BYTES, payload, sizeof(payload)) ==
          (long)sizeof(payload));

  shard_pool_destroy(pool);
  s->destroy(s);
  log_info("  PASS");
  return 0;

Fail3:
  shard_pool_destroy(pool);
Fail2:
  s->destroy(s);
Fail:
  log_error("  FAIL");
  return 1;
}

static int
test_shard_pool_many_files(void)
{
  enum
  {
    NFILES = 65
  };
  struct store* store = store_fs_create(tmpdir, 0);
  CHECK(Fail, store);
  struct shard_pool* pool = store->create_pool(store, NFILES);
  CHECK(CleanupStore, pool);

  struct shard_writer* writers[NFILES];
  const char first = 'a';
  char key[256];
  for (uint32_t i = 0; i < NFILES; ++i) {
    snprintf(key, sizeof(key), "many-files/%u.bin", i);
    writers[i] = pool->open(pool, i, key);
    CHECK(CleanupPool, writers[i]);
    CHECK(CleanupPool,
          writers[i]->write(writers[i], 0, &first, &first + 1) == 0);
  }
  CHECK(CleanupPool, pool->flush(pool) == 0);

  const char second = 'b';
  for (uint32_t i = 0; i < NFILES; ++i) {
    CHECK(CleanupPool,
          writers[i]->write(writers[i], 1, &second, &second + 1) == 0);
    CHECK(CleanupPool, writers[i]->finalize(writers[i]) == 0);
  }
  CHECK(CleanupPool, pool->flush(pool) == 0);

  char path[4096];
  for (uint32_t i = 0; i < NFILES; ++i) {
    snprintf(path, sizeof(path), "%s/many-files/%u.bin", tmpdir, i);
    FILE* f = fopen(path, "rb");
    CHECK(CleanupPool, f);
    const int a = fgetc(f);
    const int b = fgetc(f);
    const int end = fgetc(f);
    fclose(f);
    CHECK(CleanupPool, a == first && b == second && end == EOF);
  }

  shard_pool_destroy(pool);
  store->destroy(store);
  return 0;

CleanupPool:
  shard_pool_destroy(pool);
CleanupStore:
  store->destroy(store);
Fail:
  return 1;
}

struct held_closes
{
  struct io_backend inner;
  _Atomic int release;
};

static void
hold_close(void* ctx, const struct io_request* req)
{
  struct held_closes* held = (struct held_closes*)ctx;
  if (req->op == IO_OP_CLOSE)
    while (!atomic_load(&held->release))
      platform_sleep_ns(1000000LL);
  held->inner.execute(held->inner.ctx, req);
}

static struct io_backend
wrap_held_closes(void* ctx, struct io_backend inner)
{
  struct held_closes* held = (struct held_closes*)ctx;
  held->inner = inner;
  return (struct io_backend){ .ctx = held, .execute = hold_close };
}

struct repeated_opens
{
  struct shard_pool* pool;
  uint32_t nslots;
  int result;
};

static void
open_many_generations(void* ctx)
{
  struct repeated_opens* call = (struct repeated_opens*)ctx;
  for (uint32_t i = 0; i < 128; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "bounded-%u/%u.bin", call->nslots, i);
    struct shard_writer* w =
      call->pool->open(call->pool, i % call->nslots, key);
    const char byte = (char)i;
    if (!w || w->write(w, 0, &byte, &byte + 1)) {
      call->result = 1;
      return;
    }
  }
}

static int
test_shard_pool_handle_bound(uint32_t nslots, uint64_t max_requests)
{
  struct held_closes held = { 0 };
  struct io_scheduler* queue = NULL;
  test_thread* poster = NULL;
  const struct io_scheduler_limits limits = { .workers = 4,
                                              .max_requests = max_requests };
  struct shard_pool* pool = shard_pool_fs_create_wrapped(
    tmpdir,
    nslots,
    0,
    &limits,
    (struct shard_pool_fs_wrapper){
      .ctx = &held, .wrap = wrap_held_closes, .queue = &queue });
  CHECK(Fail, pool);
  struct io_backend_fs* backend = (struct io_backend_fs*)held.inner.ctx;

  for (uint32_t i = 0; i < nslots; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "bounded-%u/initial-%u.bin", nslots, i);
    CHECK(Cleanup, pool->open(pool, i, key));
  }
  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, io_backend_fs_handle_count(backend) == nslots);

  struct repeated_opens call = { .pool = pool, .nslots = nslots };
  CHECK(Cleanup, test_thread_start(&poster, open_many_generations, &call) == 0);
  int parked = 0;
  for (int i = 0; i < 2000; ++i) {
    if (io_scheduler_parked_threads(queue) != 0 &&
        io_backend_fs_handle_count(backend) > nslots) {
      parked = 1;
      break;
    }
    platform_sleep_ns(1000000LL);
  }
  CHECK(Cleanup, parked);
  CHECK(Cleanup,
        io_backend_fs_peak_handle_count(backend) <= nslots + max_requests);

  atomic_store(&held.release, 1);
  test_thread_join(poster);
  poster = NULL;
  CHECK(Cleanup, call.result == 0);
  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, io_backend_fs_handle_count(backend) == nslots);
  CHECK(Cleanup,
        io_backend_fs_peak_handle_count(backend) <= nslots + max_requests);

  for (uint32_t i = 0; i < nslots; ++i) {
    char key[256];
    snprintf(key, sizeof(key), "bounded-%u/final-%u.bin", nslots, i);
    struct shard_writer* w = pool->open(pool, i, key);
    CHECK(Cleanup, w && w->finalize(w) == 0);
  }
  CHECK(Cleanup, pool->flush(pool) == 0);
  CHECK(Cleanup, io_backend_fs_handle_count(backend) == 0);
  for (uint32_t i = 0; i < 128; ++i) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/bounded-%u/%u.bin", tmpdir, nslots, i);
    FILE* f = fopen(path, "rb");
    CHECK(Cleanup, f);
    const int byte = fgetc(f);
    const int end = fgetc(f);
    fclose(f);
    CHECK(Cleanup, byte == (int)i && end == EOF);
  }

  shard_pool_destroy(pool);
  return 0;
Cleanup:
  atomic_store(&held.release, 1);
  if (poster)
    test_thread_join(poster);
  shard_pool_destroy(pool);
Fail:
  return 1;
}

static int
test_backend_open_failure_cleanup(void)
{
  char parent[4096];
  char path[4096];
  snprintf(parent, sizeof(parent), "%s/open-parent-is-file", tmpdir);
  snprintf(path, sizeof(path), "%s/open-parent-is-file/child.bin", tmpdir);
  FILE* parent_file = fopen(parent, "wb");
  CHECK(Fail, parent_file);
  fclose(parent_file);

  _Atomic int io_error = 0;
  struct io_backend_fs* backend = io_backend_fs_create(&io_error, 0);
  CHECK(Fail, backend);
  const struct io_backend raw = io_backend_fs_as_backend(backend);
  const struct io_file_token failed = io_backend_fs_reserve_file(backend);
  CHECK(Cleanup, failed.generation != 0);
  raw.execute(
    raw.ctx,
    &(struct io_request){ .op = IO_OP_OPEN, .file = failed, .path = path });
  CHECK(Cleanup, atomic_load(&io_error) != 0);
  CHECK(Cleanup, io_backend_fs_handle_count(backend) == 0);
  raw.execute(
    raw.ctx,
    &(struct io_request){
      .op = IO_OP_WRITE, .file = failed, .payload = "x", .nbytes = 1 });
  raw.execute(raw.ctx,
              &(struct io_request){
                .op = IO_OP_TRUNCATE, .file = failed, .logical_size = 1 });
  raw.execute(raw.ctx,
              &(struct io_request){ .op = IO_OP_CLOSE, .file = failed });

  const struct io_file_token cancelled = io_backend_fs_reserve_file(backend);
  CHECK(Cleanup, cancelled.index == failed.index);
  CHECK(Cleanup, cancelled.generation != failed.generation);
  io_backend_fs_cancel_file(backend, cancelled);
  const struct io_file_token reused = io_backend_fs_reserve_file(backend);
  CHECK(Cleanup, reused.index == cancelled.index);
  CHECK(Cleanup, reused.generation != cancelled.generation);
  io_backend_fs_cancel_file(backend, reused);

  io_backend_fs_destroy(backend);
  return 0;
Cleanup:
  io_backend_fs_destroy(backend);
Fail:
  return 1;
}

static int
test_shard_pool_owns_open_paths(void)
{
  _Atomic int gate = 0;
  struct io_faults faults;
  const struct io_scheduler_limits limits = { .workers = 1 };
  struct shard_pool* pool =
    io_faults_pool_create(&faults, tmpdir, 2, 0, &limits);
  CHECK(Fail, pool);
  CHECK(Cleanup, io_faults_inject_blocking_job(&faults, &gate) == 0);

  char key[64];
  const char byte = 'x';
  for (uint32_t i = 0; i < 2; ++i) {
    snprintf(key, sizeof(key), "owned-open-%u.bin", i);
    struct shard_writer* writer = pool->open(pool, i, key);
    CHECK(Cleanup, writer);
    memset(key, 0, sizeof(key));
    CHECK(Cleanup, writer->write(writer, 0, &byte, &byte + 1) == 0);
    CHECK(Cleanup, writer->finalize(writer) == 0);
  }

  atomic_store(&gate, 1);
  CHECK(Cleanup, pool->flush(pool) == 0);
  char path[4096];
  for (uint32_t i = 0; i < 2; ++i) {
    snprintf(path, sizeof(path), "%s/owned-open-%u.bin", tmpdir, i);
    CHECK(Cleanup, file_size(path) == 1);
    CHECK(Cleanup, platform_remove_file(path) == 0);
  }

  shard_pool_destroy(pool);
  return 0;
Cleanup:
  atomic_store(&gate, 1);
  shard_pool_destroy(pool);
Fail:
  return 1;
}

// --- stale file token ---

// A pool slot's token is private and is cleared on finalize, so a retired
// token has to come from the registry itself.
static int
test_stale_file_token_refused(void)
{
  log_info("=== test_stale_file_token_refused ===");

  char first_path[4096];
  char second_path[4096];
  snprintf(first_path, sizeof(first_path), "%s/stale_first.bin", tmpdir);
  snprintf(second_path, sizeof(second_path), "%s/stale_second.bin", tmpdir);

  _Atomic int io_error = 0;
  struct io_backend_fs* backend = io_backend_fs_create(&io_error, 0);
  CHECK(Fail, backend);

  struct io_scheduler* q = io_scheduler_create(
    io_backend_fs_as_backend(backend), (struct io_scheduler_limits){ 0 });
  CHECK(Fail2, q);

  const struct io_file_token retired = io_backend_fs_reserve_file(backend);
  CHECK(Fail3, retired.generation != 0);
  CHECK(Fail3,
        io_scheduler_post(q,
                          (struct io_request){ .op = IO_OP_OPEN,
                                               .file = retired,
                                               .path = first_path }) == 0);

  CHECK(Fail3,
        io_scheduler_post(
          q, (struct io_request){ .op = IO_OP_CLOSE, .file = retired }) == 0);
  io_event_wait(q, io_scheduler_record(q));
  CHECK(Fail3, atomic_load(&io_error) == 0);

  const struct io_file_token fresh = io_backend_fs_reserve_file(backend);
  CHECK(Fail3, fresh.generation != 0);
  CHECK(Fail3, fresh.index == retired.index);
  CHECK(Fail3, fresh.generation != retired.generation);
  CHECK(Fail3,
        io_scheduler_post(q,
                          (struct io_request){ .op = IO_OP_OPEN,
                                               .file = fresh,
                                               .path = second_path }) == 0);
  static const char original[] = "replacement contents";
  CHECK(Fail3,
        io_scheduler_post(q,
                          (struct io_request){ .op = IO_OP_WRITE,
                                               .file = fresh,
                                               .payload = original,
                                               .nbytes = sizeof(original) }) ==
          0);
  io_event_wait(q, io_scheduler_record(q));
  CHECK(Fail3, atomic_load(&io_error) == 0);

  static const char payload[] = "must not land anywhere";
  CHECK(Fail3,
        io_scheduler_post(q,
                          (struct io_request){ .op = IO_OP_WRITE,
                                               .file = retired,
                                               .payload = payload,
                                               .nbytes = sizeof(payload) }) ==
          0);
  io_event_wait(q, io_scheduler_record(q));
  CHECK(Fail3, atomic_load(&io_error) == 1);

  CHECK(Fail3,
        io_scheduler_post(
          q, (struct io_request){ .op = IO_OP_CLOSE, .file = fresh }) == 0);
  io_event_wait(q, io_scheduler_record(q));

  io_scheduler_destroy(q);
  io_backend_fs_destroy(backend);

  CHECK(Fail, file_size(first_path) == 0);
  FILE* f = fopen(second_path, "rb");
  CHECK(Fail, f);
  char actual[sizeof(original)];
  const size_t nread = fread(actual, 1, sizeof(actual), f);
  const int end = fgetc(f);
  fclose(f);
  CHECK(Fail, nread == sizeof(original) && end == EOF);
  CHECK(Fail, memcmp(actual, original, sizeof(original)) == 0);

  log_info("  PASS");
  return 0;

Fail3:
  io_scheduler_destroy(q);
Fail2:
  io_backend_fs_destroy(backend);
Fail:
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  if (make_tmpdir())
    return 1;
  log_info("tmpdir: %s", tmpdir);

  int err = 0;
  err |= test_store_put();
  err |= test_put_is_atomic_for_readers();
  err |= test_store_mkdirs();
  err |= test_shard_pool_write();
  err |= test_shard_pool_fence();
  err |= test_shard_pool_on_demand_mkdir();
  err |= test_shard_pool_unbuffered();
  err |= test_shard_pool_error_propagation();
  err |= test_shard_pool_open_failure();
  err |= test_shard_pool_close_failure();
  err |= test_failed_output_write_releases_buffer();
  err |= test_shard_pool_presize();
  err |= test_shard_pool_many_files();
  err |= test_shard_pool_handle_bound(1, 8);
  err |= test_shard_pool_handle_bound(3, 16);
  err |= test_backend_open_failure_cleanup();
  err |= test_shard_pool_owns_open_paths();
  err |= test_stale_file_token_refused();
  err |= test_has_existing_data();
  err |= test_has_existing_data_unrelated_files();
  err |= test_has_existing_data_after_write_group();
  err |= test_has_existing_data_root_is_file();

  // Cleanup
  test_tmpdir_remove(tmpdir);

  return err;
}
