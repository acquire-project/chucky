#pragma once

#include <stddef.h>
#include <stdint.h>

struct json_writer
{
  char* buf;
  size_t cap;
  size_t pos;
  int needs_comma;
  int error;
};

void
jw_init(struct json_writer* jw, char* buf, size_t cap);
void
jw_object_begin(struct json_writer* jw);
void
jw_object_end(struct json_writer* jw);
void
jw_array_begin(struct json_writer* jw);
void
jw_array_end(struct json_writer* jw);
void
jw_key(struct json_writer* jw, const char* key);
void
jw_string(struct json_writer* jw, const char* val);
void
jw_int(struct json_writer* jw, int64_t val);
void
jw_uint(struct json_writer* jw, uint64_t val);
void
jw_float(struct json_writer* jw, double val);
void
jw_null(struct json_writer* jw);
void
jw_bool(struct json_writer* jw, int val);
size_t
jw_length(const struct json_writer* jw);
int
jw_error(const struct json_writer* jw);
