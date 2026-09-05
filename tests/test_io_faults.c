#include "test_io_faults.h"
#include "platform/platform.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/io_scheduler.h"
#include "zarr/shard_pool_fs.h"
#include "zarr/store.h"
#include "zarr/store_fs.h"

#include <stdlib.h>
#include <string.h>

static void
faults_execute(void* ctx, const struct io_request* req)
{
  struct io_faults* f = (struct io_faults*)ctx;

  uint16_t armed = atomic_load(&f->armed);
  uint8_t fault = IO_FAULT_NONE;
  if ((armed >> 8) == req->op &&
      atomic_compare_exchange_strong(&f->armed, &armed, (uint16_t)0))
    fault = (uint8_t)(armed & 0xff);

  if (fault == IO_FAULT_FAIL) {
    log_error("io_faults: injected failure of io op %u", (unsigned)req->op);
    shard_pool_fs_set_error(f->pool);
    return;
  }

  if (fault == IO_FAULT_BLOCK) {
    while (atomic_load(f->block_gate) == 0)
      platform_sleep_ns(1000000LL);
  }

  f->inner.execute(f->inner.ctx, req);
}

static struct io_backend
faults_wrap(void* ctx, struct io_backend inner)
{
  struct io_faults* f = (struct io_faults*)ctx;
  f->inner = inner;
  return (struct io_backend){ .ctx = f, .execute = faults_execute };
}

static struct shard_pool*
pool_create(struct io_faults* f,
            const char* root,
            uint64_t nslots,
            int unbuffered,
            const struct io_scheduler_limits* limits)
{
  f->pool = shard_pool_fs_create_wrapped(root,
                                         nslots,
                                         unbuffered,
                                         limits,
                                         (struct shard_pool_fs_wrapper){
                                           .ctx = f,
                                           .wrap = faults_wrap,
                                           .queue = &f->queue,
                                         });
  return f->pool;
}

struct shard_pool*
io_faults_pool_create(struct io_faults* f,
                      const char* root,
                      uint64_t nslots,
                      int unbuffered,
                      const struct io_scheduler_limits* limits)
{
  memset(f, 0, sizeof(*f));
  return pool_create(f, root, nslots, unbuffered, limits);
}

// --- Store ---

struct faults_store
{
  struct store base;
  struct store* inner; // owned
  struct io_faults* faults;
  struct strbuf root; // owned
  int unbuffered;
  struct io_scheduler_limits limits;
};

static int
faults_store_put(struct store* self,
                 const char* key,
                 const void* data,
                 size_t len)
{
  struct faults_store* s = container_of(self, struct faults_store, base);
  return s->inner->put(s->inner, key, data, len);
}

static int
faults_store_mkdirs(struct store* self, const char* key)
{
  struct faults_store* s = container_of(self, struct faults_store, base);
  return s->inner->mkdirs(s->inner, key);
}

static int
faults_store_has_existing_data(struct store* self)
{
  struct faults_store* s = container_of(self, struct faults_store, base);
  return s->inner->has_existing_data(s->inner);
}

static struct shard_pool*
faults_store_create_pool(struct store* self, uint64_t nslots)
{
  struct faults_store* s = container_of(self, struct faults_store, base);
  // A second pool would leave the first pool's queue calling the wrong
  // backend.
  CHECK(Fail, !s->faults->pool);
  return pool_create(
    s->faults, strbuf_cstr(&s->root), nslots, s->unbuffered, &s->limits);
Fail:
  return NULL;
}

static void
faults_store_destroy(struct store* self)
{
  struct faults_store* s = container_of(self, struct faults_store, base);
  s->inner->destroy(s->inner);
  strbuf_free(&s->root);
  free(s);
}

struct store*
io_faults_store_create(struct io_faults* f,
                       const char* root,
                       int unbuffered,
                       const struct io_scheduler_limits* limits)
{
  memset(f, 0, sizeof(*f));

  struct faults_store* s = (struct faults_store*)calloc(1, sizeof(*s));
  CHECK(Fail, s);

  s->base.put = faults_store_put;
  s->base.mkdirs = faults_store_mkdirs;
  s->base.create_pool = faults_store_create_pool;
  s->base.has_existing_data = faults_store_has_existing_data;
  s->base.destroy = faults_store_destroy;
  s->faults = f;
  s->unbuffered = unbuffered;
  if (limits)
    s->limits = *limits;
  CHECK(Fail_alloc, strbuf_set(&s->root, root) == 0);

  s->inner = store_fs_create(root, unbuffered);
  CHECK(Fail_alloc, s->inner);

  return &s->base;

Fail_alloc:
  strbuf_free(&s->root);
  free(s);
Fail:
  return NULL;
}

// --- Injection ---

static void
arm_fault(struct io_faults* f, uint8_t op, uint8_t fault)
{
  atomic_store(&f->armed, (uint16_t)(((uint16_t)op << 8) | fault));
}

static int
post_noop(struct io_faults* f)
{
  return io_scheduler_post(f->queue, (struct io_request){ .op = IO_OP_NOOP });
}

int
io_faults_inject_failing_job(struct io_faults* f)
{
  arm_fault(f, IO_OP_NOOP, IO_FAULT_FAIL);
  return post_noop(f);
}

int
io_faults_inject_blocking_job(struct io_faults* f, _Atomic int* gate)
{
  f->block_gate = gate;
  arm_fault(f, IO_OP_NOOP, IO_FAULT_BLOCK);
  return post_noop(f);
}

void
io_faults_fail_next_truncate(struct io_faults* f)
{
  arm_fault(f, IO_OP_TRUNCATE, IO_FAULT_FAIL);
}

void
io_faults_fail_next_write(struct io_faults* f)
{
  arm_fault(f, IO_OP_WRITE, IO_FAULT_FAIL);
}

void
io_faults_fail_next_open(struct io_faults* f)
{
  arm_fault(f, IO_OP_OPEN, IO_FAULT_FAIL);
}

void
io_faults_fail_next_close(struct io_faults* f)
{
  arm_fault(f, IO_OP_CLOSE, IO_FAULT_FAIL);
}
