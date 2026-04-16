// Buffered set of (attr_key, json_value) pairs used by zarr/ngff/hcs handles
// to collect custom metadata attributes until the next zarr.json rewrite.
#pragma once

#include <stddef.h>

struct attr_pair
{
  char* key;
  char* json_value;
};

struct attr_set
{
  struct attr_pair* items;
  size_t count;
  size_t cap;
  int dirty;
};

struct json_writer;

void
attr_set_init(struct attr_set* s);

void
attr_set_destroy(struct attr_set* s);

// Validate key and json_value; upsert (replace if present, else append).
// Marks the set dirty on success. Returns 0 on success.
int
attr_set_upsert(struct attr_set* s,
                const char* attr_key,
                const char* json_value);

// Emit each pair as "key":<raw_value> into jw, comma-separated.
// Caller is inside an object and places pairs alongside other keys.
void
attr_set_emit(const struct attr_set* s, struct json_writer* jw);

// True iff the key is a valid attribute key (no control, quotes, backslash,
// and non-empty).
int
attr_set_key_ok(const char* attr_key);
