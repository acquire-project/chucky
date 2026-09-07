#include "zarr/metadata_io.h"
#include "util/strbuf.h"
#include "zarr/shard_pool.h"
#include "zarr/store.h"

int
zarr_metadata_submit(struct store* store,
                     struct shard_pool* pool,
                     const char* prefix,
                     const struct strbuf* json)
{
  struct strbuf key = { 0 };
  int rc = prefix && prefix[0] ? strbuf_appendf(&key, "%s/zarr.json", prefix)
                               : strbuf_append_cstr(&key, "zarr.json");
  if (rc == 0) {
    if (pool && pool->queue_metadata)
      rc = pool->queue_metadata(
        pool, strbuf_cstr(&key), strbuf_cstr(json), strbuf_len(json));
    else
      rc = store->put(
        store, strbuf_cstr(&key), strbuf_cstr(json), strbuf_len(json));
  }
  strbuf_free(&key);
  return rc;
}

int
zarr_metadata_wait(struct shard_pool* pool)
{
  return pool && pool->queue_metadata ? pool->flush(pool) : 0;
}
