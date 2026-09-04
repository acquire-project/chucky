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

  _Atomic int io_error;
  atomic_store(&io_error, 0);

  struct io_backend_fs* backend = io_backend_fs_create(&io_error);
  CHECK(Fail, backend);

  struct io_scheduler* q = io_scheduler_create(
    io_backend_fs_as_backend(backend), (struct io_scheduler_limits){ 0 });
  CHECK(Fail2, q);

  platform_fd fd = platform_open_write(first_path, 0);
  CHECK(Fail3, fd != PLATFORM_FD_INVALID);
  const struct io_file_token retired = io_backend_fs_add_file(backend, fd);
  CHECK(Fail3, retired.generation != 0);

  CHECK(Fail3,
        io_scheduler_post(
          q, (struct io_request){ .op = IO_OP_CLOSE, .file = retired }) == 0);
  io_event_wait(q, io_scheduler_record(q));
  CHECK(Fail3, atomic_load(&io_error) == 0);

  // The slot freed by the close is reused under a new generation.
  platform_fd second_fd = platform_open_write(second_path, 0);
  CHECK(Fail3, second_fd != PLATFORM_FD_INVALID);
  const struct io_file_token fresh = io_backend_fs_add_file(backend, second_fd);
  CHECK(Fail3, fresh.generation != 0);
  CHECK(Fail3, fresh.index == retired.index);
  CHECK(Fail3, fresh.generation != retired.generation);

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
  CHECK(Fail, file_size(second_path) == 0);

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
  err |= test_failed_output_write_releases_buffer();
  err |= test_shard_pool_presize();
  err |= test_stale_file_token_refused();
  err |= test_has_existing_data();
  err |= test_has_existing_data_unrelated_files();
  err |= test_has_existing_data_after_write_group();
  err |= test_has_existing_data_root_is_file();

  // Cleanup
  test_tmpdir_remove(tmpdir);

  return err;
}
