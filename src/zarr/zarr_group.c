#include "zarr/zarr_group.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr.h"
#include "zarr/attr_set.h"
#include "zarr/json_writer.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>

int
zarr_group_write_with_raw_attrs(struct store* store,
                                const char* key,
                                const char* attributes_json)
{
  CHECK(Fail, store);
  CHECK(Fail, key);
  CHECK(Fail, attributes_json);

  struct strbuf buf = { 0 };
  int rc = strbuf_appendf(&buf,
                          "{\"zarr_format\":3,\"node_type\":\"group\","
                          "\"consolidated_metadata\":null,\"attributes\":%s}",
                          attributes_json);
  if (rc == 0)
    rc = store->put(store, key, strbuf_cstr(&buf), strbuf_len(&buf));
  strbuf_free(&buf);
  return rc;

Fail:
  return 1;
}

// --- Handle-based group with buffered attributes ---

struct zarr_group
{
  struct store* store; // borrowed
  struct strbuf key;   // owned
  struct attr_set attrs;
};

static int
zarr_group_write(struct zarr_group* g)
{
  struct strbuf json = { 0 };
  int rc = 1;

  struct json_writer jw;
  jw_init(&jw, &json);

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_key(&jw, "consolidated_metadata");
  jw_null(&jw);
  jw_key(&jw, "attributes");
  jw_object_begin(&jw);
  attr_set_emit(&g->attrs, &jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  if (jw_error(&jw))
    goto done;

  rc = g->store->put(
    g->store, strbuf_cstr(&g->key), strbuf_cstr(&json), strbuf_len(&json));
  if (rc == 0)
    g->attrs.dirty = 0;

done:
  strbuf_free(&json);
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
  int rc = (key[0]) ? strbuf_appendf(&g->key, "%s/zarr.json", key)
                    : strbuf_append_cstr(&g->key, "zarr.json");
  if (rc || zarr_group_write(g) != 0) {
    attr_set_destroy(&g->attrs);
    strbuf_free(&g->key);
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
  strbuf_free(&g->key);
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
