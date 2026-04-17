#include "zarr/json_validate.h"

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
  if (pos < len && s[pos] == '.') {
    ++pos;
    size_t frac_start = pos;
    while (pos < len && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
    if (pos == frac_start)
      return -1;
  }
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

static ptrdiff_t
skip_value(const char* s, size_t len, size_t pos)
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
      ptrdiff_t np = skip_value(s, len, pos);
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
      np = skip_value(s, len, pos);
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
json_value_is_valid(const char* s, size_t len)
{
  if (!s)
    return 0;
  ptrdiff_t end = skip_value(s, len, 0);
  if (end < 0)
    return 0;
  size_t pos = skip_ws(s, len, (size_t)end);
  return pos == len ? 1 : 0;
}
