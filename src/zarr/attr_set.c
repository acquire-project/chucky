#include "zarr/attr_set.h"
#include "util/prelude.h"
#include "zarr/json_validate.h"
#include "zarr/json_writer.h"

#include <stdlib.h>
#include <string.h>

void
attr_set_init(struct attr_set* s)
{
  s->items = NULL;
  s->count = 0;
  s->cap = 0;
  s->dirty = 0;
}

void
attr_set_destroy(struct attr_set* s)
{
  if (!s)
    return;
  for (size_t i = 0; i < s->count; ++i) {
    free(s->items[i].key);
    free(s->items[i].json_value);
  }
  free(s->items);
  s->items = NULL;
  s->count = 0;
  s->cap = 0;
  s->dirty = 0;
}

int
attr_set_key_ok(const char* attr_key)
{
  if (!attr_key || !attr_key[0])
    return 0;
  for (const char* p = attr_key; *p; ++p) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x20)
      return 0;
    if (c == '"' || c == '\\')
      return 0;
  }
  return 1;
}

static char*
dup_cstr(const char* s)
{
  size_t n = strlen(s);
  char* out = (char*)malloc(n + 1);
  if (!out)
    return NULL;
  memcpy(out, s, n + 1);
  return out;
}

int
attr_set_upsert(struct attr_set* s,
                const char* attr_key,
                const char* json_value)
{
  if (!s || !attr_set_key_ok(attr_key) || !json_value)
    return 1;
  if (!json_value_is_valid(json_value, strlen(json_value)))
    return 1;

  // Replace if present.
  for (size_t i = 0; i < s->count; ++i) {
    if (strcmp(s->items[i].key, attr_key) == 0) {
      char* nv = dup_cstr(json_value);
      if (!nv)
        return 1;
      free(s->items[i].json_value);
      s->items[i].json_value = nv;
      s->dirty = 1;
      return 0;
    }
  }

  // Append.
  if (s->count == s->cap) {
    size_t ncap = s->cap ? s->cap * 2 : 4;
    struct attr_pair* ni =
      (struct attr_pair*)realloc(s->items, ncap * sizeof(*ni));
    if (!ni)
      return 1;
    s->items = ni;
    s->cap = ncap;
  }
  char* nk = dup_cstr(attr_key);
  char* nv = dup_cstr(json_value);
  if (!nk || !nv) {
    free(nk);
    free(nv);
    return 1;
  }
  s->items[s->count].key = nk;
  s->items[s->count].json_value = nv;
  ++s->count;
  s->dirty = 1;
  return 0;
}

void
attr_set_emit(const struct attr_set* s, struct json_writer* jw)
{
  if (!s)
    return;
  for (size_t i = 0; i < s->count; ++i) {
    jw_key(jw, s->items[i].key);
    jw_raw(jw, s->items[i].json_value);
  }
}
