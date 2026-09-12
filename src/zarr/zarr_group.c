#include "zarr/zarr_group.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr.h"
#include "zarr/attr_set.h"
#include "zarr/json_writer.h"
#include "zarr/metadata_io.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>

int
zarr_group_submit(struct store* store,
                  struct shard_pool* pool,
                  const char* prefix,
                  const char* attributes_json)
{
  CHECK(Fail, store);
  CHECK(Fail, prefix);
  CHECK(Fail, attributes_json);

  struct strbuf buf = { 0 };
  int rc = zarr_group_json(&buf, attributes_json);
  if (rc == 0)
    rc = zarr_metadata_submit(store, pool, prefix, &buf);
  strbuf_free(&buf);
  return rc;

Fail:
  return 1;
}

// --- Handle-based group with buffered attributes ---

struct zarr_group
{
  struct store* store;  // borrowed
  struct strbuf prefix; // owned
  struct attr_set attrs;
};

static int
zarr_group_write(struct zarr_group* g)
{
  struct strbuf attrs = { 0 };
  int rc = 1;

  struct json_writer jw;
  jw_init(&jw, &attrs);

  jw_object_begin(&jw);
  attr_set_emit(&g->attrs, &jw);
  jw_object_end(&jw);

  if (jw_error(&jw))
    goto done;

  rc = zarr_group_submit(
    g->store, NULL, strbuf_cstr(&g->prefix), strbuf_cstr(&attrs));
  if (rc == 0)
    g->attrs.dirty = 0;

done:
  strbuf_free(&attrs);
  return rc;
}

struct zarr_group*
zarr_group_create(struct store* store, const char* key)
{
  CHECK(Fail, store);
  CHECK(Fail, key);

  struct zarr_group* g = (struct zarr_group*)calloc(1, sizeof(*g));
  CHECK(Fail, g);
  g->store = store;
  attr_set_init(&g->attrs);
  int rc = strbuf_set(&g->prefix, key);
  if (rc || zarr_group_write(g) != 0) {
    attr_set_destroy(&g->attrs);
    strbuf_free(&g->prefix);
    free(g);
    return NULL;
  }
  return g;

Fail:
  return NULL;
}

void
zarr_group_destroy(struct zarr_group* g)
{
  if (!g)
    return;
  if (g->attrs.dirty)
    zarr_group_write(g);
  attr_set_destroy(&g->attrs);
  strbuf_free(&g->prefix);
  free(g);
}

int
zarr_group_set_attribute(struct zarr_group* g,
                         const char* attr_key,
                         const char* json_value)
{
  CHECK(Fail, g);
  return attr_set_upsert(&g->attrs, attr_key, json_value);
Fail:
  return 1;
}

int
zarr_group_flush_metadata(struct zarr_group* g)
{
  CHECK(Fail, g);
  if (!g->attrs.dirty)
    return 0;
  return zarr_group_write(g);
Fail:
  return 1;
}
