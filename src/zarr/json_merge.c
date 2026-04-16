#include "zarr/json_merge.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

static size_t
skip_ws(const char* s, size_t len, size_t pos)
{
  while (pos < len) {
    char c = s[pos];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
      ++pos;
    else
      break;
  }
  return pos;
}

static ptrdiff_t
skip_string(const char* s, size_t len, size_t pos)
{
  if (pos >= len || s[pos] != '"')
    return -1;
  ++pos;
  while (pos < len) {
    char c = s[pos];
    if (c == '\\') {
      if (pos + 1 >= len)
        return -1;
      pos += 2;
      continue;
    }
    if (c == '"')
      return (ptrdiff_t)(pos + 1);
    // reject unescaped control chars (< 0x20)
    if ((unsigned char)c < 0x20)
      return -1;
    ++pos;
  }
  return -1;
}

static ptrdiff_t
skip_number(const char* s, size_t len, size_t pos)
{
  size_t start = pos;
  if (pos < len && s[pos] == '-')
    ++pos;
  // integer part
  if (pos >= len)
    return -1;
  if (s[pos] == '0') {
    ++pos;
  } else if (s[pos] >= '1' && s[pos] <= '9') {
    ++pos;
    while (pos < len && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
  } else {
    return -1;
  }
  // fraction
  if (pos < len && s[pos] == '.') {
    ++pos;
    size_t frac_start = pos;
    while (pos < len && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
    if (pos == frac_start)
      return -1;
  }
  // exponent
  if (pos < len && (s[pos] == 'e' || s[pos] == 'E')) {
    ++pos;
    if (pos < len && (s[pos] == '+' || s[pos] == '-'))
      ++pos;
    size_t exp_start = pos;
    while (pos < len && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
    if (pos == exp_start)
      return -1;
  }
  if (pos == start)
    return -1;
  return (ptrdiff_t)pos;
}

static ptrdiff_t
skip_literal(const char* s, size_t len, size_t pos, const char* lit)
{
  size_t llen = strlen(lit);
  if (pos + llen > len)
    return -1;
  if (memcmp(s + pos, lit, llen) != 0)
    return -1;
  return (ptrdiff_t)(pos + llen);
}

ptrdiff_t
json_skip_value(const char* s, size_t len, size_t pos)
{
  pos = skip_ws(s, len, pos);
  if (pos >= len)
    return -1;

  char c = s[pos];
  if (c == '"')
    return skip_string(s, len, pos);
  if (c == 't')
    return skip_literal(s, len, pos, "true");
  if (c == 'f')
    return skip_literal(s, len, pos, "false");
  if (c == 'n')
    return skip_literal(s, len, pos, "null");
  if (c == '-' || (c >= '0' && c <= '9'))
    return skip_number(s, len, pos);

  if (c == '[') {
    ++pos;
    pos = skip_ws(s, len, pos);
    if (pos < len && s[pos] == ']')
      return (ptrdiff_t)(pos + 1);
    for (;;) {
      ptrdiff_t np = json_skip_value(s, len, pos);
      if (np < 0)
        return -1;
      pos = (size_t)np;
      pos = skip_ws(s, len, pos);
      if (pos >= len)
        return -1;
      if (s[pos] == ']')
        return (ptrdiff_t)(pos + 1);
      if (s[pos] != ',')
        return -1;
      ++pos;
    }
  }

  if (c == '{') {
    ++pos;
    pos = skip_ws(s, len, pos);
    if (pos < len && s[pos] == '}')
      return (ptrdiff_t)(pos + 1);
    for (;;) {
      pos = skip_ws(s, len, pos);
      ptrdiff_t np = skip_string(s, len, pos);
      if (np < 0)
        return -1;
      pos = (size_t)np;
      pos = skip_ws(s, len, pos);
      if (pos >= len || s[pos] != ':')
        return -1;
      ++pos;
      np = json_skip_value(s, len, pos);
      if (np < 0)
        return -1;
      pos = (size_t)np;
      pos = skip_ws(s, len, pos);
      if (pos >= len)
        return -1;
      if (s[pos] == '}')
        return (ptrdiff_t)(pos + 1);
      if (s[pos] != ',')
        return -1;
      ++pos;
    }
  }

  return -1;
}

int
json_validate(const char* s, size_t len)
{
  ptrdiff_t end = json_skip_value(s, len, 0);
  if (end < 0)
    return 1;
  size_t pos = skip_ws(s, len, (size_t)end);
  return pos == len ? 0 : 1;
}

// Locate "attributes" key inside the top-level object of zjson.
// Returns 0 on success, non-zero on failure.
// On success: *attrs_start and *attrs_end bound the attributes object value
// (including the braces).
static int
find_attributes(const char* zjson,
                size_t len,
                size_t* attrs_start,
                size_t* attrs_end)
{
  size_t pos = skip_ws(zjson, len, 0);
  if (pos >= len || zjson[pos] != '{')
    return 1;
  ++pos;

  for (;;) {
    pos = skip_ws(zjson, len, pos);
    if (pos >= len)
      return 1;
    if (zjson[pos] == '}')
      return 1; // no attributes key found

    // key
    ptrdiff_t key_end = skip_string(zjson, len, pos);
    if (key_end < 0)
      return 1;

    // bare key contents without quotes
    size_t key_start = pos + 1;
    size_t key_len = (size_t)key_end - pos - 2;
    int is_attrs = (key_len == 10) &&
                   memcmp(zjson + key_start, "attributes", 10) == 0;

    pos = (size_t)key_end;
    pos = skip_ws(zjson, len, pos);
    if (pos >= len || zjson[pos] != ':')
      return 1;
    ++pos;

    pos = skip_ws(zjson, len, pos);
    size_t val_start = pos;
    ptrdiff_t val_end = json_skip_value(zjson, len, pos);
    if (val_end < 0)
      return 1;

    if (is_attrs) {
      // Must be an object.
      if (zjson[val_start] != '{')
        return 1;
      *attrs_start = val_start;
      *attrs_end = (size_t)val_end;
      return 0;
    }

    pos = (size_t)val_end;
    pos = skip_ws(zjson, len, pos);
    if (pos >= len)
      return 1;
    if (zjson[pos] == '}')
      return 1;
    if (zjson[pos] != ',')
      return 1;
    ++pos;
  }
}

// Search for attr_key inside the attributes object [attrs_start, attrs_end).
// If found: set *key_start to beginning of "key" token and *val_end to index
// just past value. Return 0.
// If not found: return 1 and set *attrs_empty to 1 if the object is empty.
static int
find_attr_key(const char* zjson,
              size_t attrs_start,
              size_t attrs_end,
              const char* attr_key,
              size_t* key_start,
              size_t* val_end,
              int* attrs_empty)
{
  size_t attr_key_len = strlen(attr_key);
  size_t pos = attrs_start;
  if (zjson[pos] != '{')
    return 1;
  ++pos;

  pos = skip_ws(zjson, attrs_end, pos);
  if (zjson[pos] == '}') {
    *attrs_empty = 1;
    return 1;
  }
  *attrs_empty = 0;

  for (;;) {
    pos = skip_ws(zjson, attrs_end, pos);
    size_t k_start_pos = pos;
    ptrdiff_t k_end = skip_string(zjson, attrs_end, pos);
    if (k_end < 0)
      return 1;

    size_t k_body = pos + 1;
    size_t k_body_len = (size_t)k_end - pos - 2;
    int match = (k_body_len == attr_key_len) &&
                memcmp(zjson + k_body, attr_key, attr_key_len) == 0;

    pos = (size_t)k_end;
    pos = skip_ws(zjson, attrs_end, pos);
    if (zjson[pos] != ':')
      return 1;
    ++pos;
    pos = skip_ws(zjson, attrs_end, pos);
    ptrdiff_t v_end = json_skip_value(zjson, attrs_end, pos);
    if (v_end < 0)
      return 1;

    if (match) {
      *key_start = k_start_pos;
      *val_end = (size_t)v_end;
      return 0;
    }

    pos = (size_t)v_end;
    pos = skip_ws(zjson, attrs_end, pos);
    if (zjson[pos] == '}')
      return 1;
    if (zjson[pos] != ',')
      return 1;
    ++pos;
  }
}

// Validate attr_key: no control chars, no ", no backslash.
static int
attr_key_ok(const char* attr_key)
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

int
json_merge_attribute(const char* zjson,
                     size_t zjson_len,
                     const char* attr_key,
                     const char* json_value,
                     char** out_buf,
                     size_t* out_len)
{
  if (!zjson || !attr_key || !json_value || !out_buf || !out_len)
    return 1;
  if (!attr_key_ok(attr_key))
    return 1;
  if (json_validate(json_value, strlen(json_value)) != 0)
    return 1;

  size_t attrs_start = 0, attrs_end = 0;
  if (find_attributes(zjson, zjson_len, &attrs_start, &attrs_end) != 0) {
    log_error("json_merge: no 'attributes' object in zarr.json");
    return 1;
  }

  size_t val_len = strlen(json_value);
  size_t attr_key_len = strlen(attr_key);

  size_t key_start = 0, val_end = 0;
  int attrs_empty = 0;
  int found = find_attr_key(zjson,
                            attrs_start,
                            attrs_end,
                            attr_key,
                            &key_start,
                            &val_end,
                            &attrs_empty) == 0;

  // Compute output.
  // Cases:
  //  A. found: replace [key_start, val_end) with "<attr_key>":<json_value>
  //  B. not found + empty attrs: change {} into {"<attr_key>":<json_value>}
  //  C. not found + non-empty: insert ,"<attr_key>":<json_value> before
  //     the closing '}' at attrs_end - 1.

  char* buf = NULL;
  size_t cap = zjson_len + attr_key_len + val_len + 32;
  buf = (char*)malloc(cap);
  if (!buf)
    return 1;
  size_t p = 0;

  if (found) {
    // Copy up to key_start
    memcpy(buf + p, zjson, key_start);
    p += key_start;
    // Insert replacement key+value
    buf[p++] = '"';
    memcpy(buf + p, attr_key, attr_key_len);
    p += attr_key_len;
    buf[p++] = '"';
    buf[p++] = ':';
    memcpy(buf + p, json_value, val_len);
    p += val_len;
    // Copy rest from val_end
    memcpy(buf + p, zjson + val_end, zjson_len - val_end);
    p += zjson_len - val_end;
  } else {
    // Insertion point: just before the closing '}' of attributes object.
    size_t close_brace = attrs_end - 1;
    if (zjson[close_brace] != '}') {
      free(buf);
      return 1;
    }

    memcpy(buf + p, zjson, close_brace);
    p += close_brace;
    if (!attrs_empty)
      buf[p++] = ',';
    buf[p++] = '"';
    memcpy(buf + p, attr_key, attr_key_len);
    p += attr_key_len;
    buf[p++] = '"';
    buf[p++] = ':';
    memcpy(buf + p, json_value, val_len);
    p += val_len;
    buf[p++] = '}';
    memcpy(buf + p, zjson + attrs_end, zjson_len - attrs_end);
    p += zjson_len - attrs_end;
  }

  *out_buf = buf;
  *out_len = p;
  return 0;
}
