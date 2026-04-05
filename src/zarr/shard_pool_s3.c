#include "zarr/shard_pool_s3.h"
#include "util/prelude.h"
#include "zarr/s3_client.h"

#include <stdlib.h>
#include <string.h>

// --- Pool struct (defined early so slot functions can access it) ---

struct s3_slot;

struct shard_pool_s3
{
  struct shard_pool base;
  struct s3_client* client; // borrowed, not owned
  char bucket[256];
  char prefix[4096]; // prepended to shard keys
  struct s3_slot* slots;
  uint64_t nslots;
  uint64_t finalize_seq;
  int finalize_err;
};

// --- S3 shard writer slot ---

struct s3_slot
{
  struct shard_writer base;
  struct shard_pool_s3* pool;
  struct s3_upload* upload;         // active upload (receiving writes)
  struct s3_upload* pending_upload; // previous upload completing async
  int pending_eof_err;              // EOF send error from finish_async
  int write_err;                    // set on part write failure
};

static int
s3_wait_pending(struct s3_slot* w)
{
  if (!w->pending_upload)
    return 0;
  int rc = w->pending_eof_err;
  rc |= s3_upload_wait(w->pending_upload);
  s3_upload_destroy(w->pending_upload);
  w->pending_upload = NULL;
  w->pending_eof_err = 0;
  if (rc)
    w->pool->finalize_err = 1;
  return rc;
}

static int
s3_slot_write(struct shard_writer* self,
              uint64_t offset,
              const void* beg,
              const void* end)
{
  (void)offset; // writes are sequential; CRT handles ordering
  struct s3_slot* w = (struct s3_slot*)self;
  if (!w->upload)
    return 1;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (nbytes == 0)
    return 0;
  if (s3_upload_write(w->upload, beg, nbytes)) {
    w->write_err = 1;
    w->pool->finalize_err = 1;
    return 1;
  }
  return 0;
}

static int
s3_slot_finalize(struct shard_writer* self)
{
  struct s3_slot* w = (struct s3_slot*)self;
  if (!w->upload)
    return 0;

  if (w->write_err) {
    s3_upload_abort(w->upload);
    w->upload = NULL;
    w->write_err = 0;
    w->pool->finalize_err = 1;
    ++w->pool->finalize_seq;
    return 1;
  }

  int eof_err = s3_upload_finish_async(w->upload);
  w->pending_upload = w->upload;
  w->pending_eof_err = eof_err;
  w->upload = NULL;
  ++w->pool->finalize_seq;
  return 0;
}

// --- Pool vtable ---

static struct shard_writer*
pool_s3_open(struct shard_pool* self, uint64_t slot, const char* key)
{
  struct shard_pool_s3* p = container_of(self, struct shard_pool_s3, base);
  CHECK(Fail, slot < p->nslots);

  struct s3_slot* w = &p->slots[slot];

  // Wait for previous upload on this slot
  s3_wait_pending(w);

  char full_key[4096];
  if (p->prefix[0])
    snprintf(full_key, sizeof(full_key), "%s/%s", p->prefix, key);
  else
    snprintf(full_key, sizeof(full_key), "%s", key);

  w->upload = s3_upload_begin(p->client, p->bucket, full_key);
  if (!w->upload) {
    log_error(
      "shard_pool_s3: failed to begin upload for %s/%s", p->bucket, full_key);
    goto Fail;
  }

  return &w->base;

Fail:
  return NULL;
}

static struct io_event
pool_s3_record_fence(struct shard_pool* self)
{
  struct shard_pool_s3* p = container_of(self, struct shard_pool_s3, base);
  return (struct io_event){ .seq = p->finalize_seq };
}

static void
pool_s3_wait_fence(struct shard_pool* self, struct io_event ev)
{
  struct shard_pool_s3* p = container_of(self, struct shard_pool_s3, base);
  if (ev.seq == 0)
    return;
  for (uint64_t i = 0; i < p->nslots; ++i)
    s3_wait_pending(&p->slots[i]);
}

static int
pool_s3_flush(struct shard_pool* self)
{
  struct shard_pool_s3* p = container_of(self, struct shard_pool_s3, base);
  for (uint64_t i = 0; i < p->nslots; ++i) {
    struct s3_slot* w = &p->slots[i];
    if (w->upload)
      s3_slot_finalize(&w->base);
    s3_wait_pending(w);
  }
  return p->finalize_err;
}

static int
pool_s3_has_error(const struct shard_pool* self)
{
  const struct shard_pool_s3* p =
    container_of(self, struct shard_pool_s3, base);
  return p->finalize_err;
}

static size_t
pool_s3_pending_bytes(const struct shard_pool* self)
{
  (void)self;
  return 0;
}

static void
pool_s3_destroy(struct shard_pool* self)
{
  struct shard_pool_s3* p = container_of(self, struct shard_pool_s3, base);

  for (uint64_t i = 0; i < p->nslots; ++i) {
    struct s3_slot* w = &p->slots[i];
    if (w->upload) {
      s3_upload_abort(w->upload);
      w->upload = NULL;
    }
    if (w->pending_upload)
      s3_wait_pending(w);
  }

  free(p->slots);
  free(p);
}

// --- Create ---

struct shard_pool*
shard_pool_s3_create(struct s3_client* client,
                     const char* bucket,
                     const char* prefix,
                     uint64_t nslots)
{
  CHECK(Fail, client);
  CHECK(Fail, bucket);
  CHECK(Fail, nslots > 0);

  struct shard_pool_s3* p =
    (struct shard_pool_s3*)calloc(1, sizeof(struct shard_pool_s3));
  CHECK(Fail, p);

  p->base.open = pool_s3_open;
  p->base.record_fence = pool_s3_record_fence;
  p->base.wait_fence = pool_s3_wait_fence;
  p->base.flush = pool_s3_flush;
  p->base.has_error = pool_s3_has_error;
  p->base.pending_bytes = pool_s3_pending_bytes;
  p->base.destroy = pool_s3_destroy;
  p->client = client;
  if (prefix)
    snprintf(p->prefix, sizeof(p->prefix), "%s", prefix);
  p->nslots = nslots;
  snprintf(p->bucket, sizeof(p->bucket), "%s", bucket);

  p->slots = (struct s3_slot*)calloc((size_t)nslots, sizeof(struct s3_slot));
  CHECK(Fail_alloc, p->slots);

  for (uint64_t i = 0; i < nslots; ++i) {
    struct s3_slot* s = &p->slots[i];
    s->base.write = s3_slot_write;
    s->base.write_direct = NULL;
    s->base.finalize = s3_slot_finalize;
    s->pool = p;
  }

  return &p->base;

Fail_alloc:
  free(p);
Fail:
  return NULL;
}
