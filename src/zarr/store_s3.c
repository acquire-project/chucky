#include "zarr/store_s3.h"
#include "util/prelude.h"
#include "zarr/s3_client.h"
#include "zarr/shard_pool_s3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct store_s3
{
  struct store base;
  struct s3_client* client;
  char bucket[256];
  char prefix[4096]; // "" if no prefix
};

// Build a full S3 key: "prefix/key" or just "key" if no prefix.
static void
s3_full_key(const struct store_s3* s, const char* key, char* out, size_t cap)
{
  if (s->prefix[0])
    snprintf(out, cap, "%s/%s", s->prefix, key);
  else
    snprintf(out, cap, "%s", key);
}

static int
s3_put(struct store* self, const char* key, const void* data, size_t len)
{
  struct store_s3* s = container_of(self, struct store_s3, base);
  char full[4096];
  s3_full_key(s, key, full, sizeof(full));
  return s3_client_put(s->client, s->bucket, full, data, len);
}

static int
s3_mkdirs(struct store* self, const char* key)
{
  (void)self;
  (void)key;
  return 0; // S3 has no directories
}

static struct shard_pool*
s3_create_pool(struct store* self, uint64_t nslots)
{
  struct store_s3* s = container_of(self, struct store_s3, base);
  return shard_pool_s3_create(s->client, s->bucket, s->prefix, nslots);
}

static void
s3_destroy(struct store* self)
{
  struct store_s3* s = container_of(self, struct store_s3, base);
  s3_client_destroy(s->client);
  free(s);
}

struct store*
store_s3_create(const struct store_s3_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->bucket);
  CHECK(Fail, cfg->region);
  CHECK(Fail, cfg->endpoint);

  struct store_s3* s = (struct store_s3*)calloc(1, sizeof(*s));
  CHECK(Fail, s);

  struct s3_client_config s3cfg = {
    .region = cfg->region,
    .endpoint = cfg->endpoint,
    .part_size = cfg->part_size,
    .throughput_gbps = cfg->throughput_gbps,
    .max_retries = cfg->max_retries,
    .backoff_scale_ms = cfg->backoff_scale_ms,
    .max_backoff_secs = cfg->max_backoff_secs,
    .timeout_ns = cfg->timeout_ns,
  };
  s->client = s3_client_create(&s3cfg);
  CHECK(Fail_alloc, s->client);

  s->base.put = s3_put;
  s->base.mkdirs = s3_mkdirs;
  s->base.create_pool = s3_create_pool;
  s->base.destroy = s3_destroy;
  snprintf(s->bucket, sizeof(s->bucket), "%s", cfg->bucket);
  if (cfg->prefix)
    snprintf(s->prefix, sizeof(s->prefix), "%s", cfg->prefix);

  return &s->base;

Fail_alloc:
  free(s);
Fail:
  return NULL;
}
