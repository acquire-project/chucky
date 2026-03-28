#include "util/prelude.h"
#include "zarr/json_writer.h"

#include <stdio.h>
#include <string.h>

static int
test_simple_object(void)
{
  char buf[256];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "hello");
  jw_key(&jw, "count");
  jw_int(&jw, 42);
  jw_key(&jw, "enabled");
  jw_bool(&jw, 1);
  jw_key(&jw, "nothing");
  jw_null(&jw);
  jw_object_end(&jw);

  const char* expected =
    "{\"name\":\"hello\",\"count\":42,\"enabled\":true,\"nothing\":null}";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) == strlen(expected));
  CHECK(Fail, memcmp(buf, expected, jw_length(&jw)) == 0);
  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

static int
test_nested(void)
{
  char buf[256];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);
  jw_key(&jw, "a");
  jw_array_begin(&jw);
  jw_int(&jw, 1);
  jw_int(&jw, 2);
  jw_int(&jw, 3);
  jw_array_end(&jw);
  jw_key(&jw, "b");
  jw_object_begin(&jw);
  jw_key(&jw, "x");
  jw_float(&jw, 3.14);
  jw_object_end(&jw);
  jw_object_end(&jw);

  const char* expected = "{\"a\":[1,2,3],\"b\":{\"x\":3.14}}";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) == strlen(expected));
  CHECK(Fail, memcmp(buf, expected, jw_length(&jw)) == 0);
  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

static int
test_string_escaping(void)
{
  char buf[256];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_string(&jw, "hello \"world\"\nnew\tline\\back\x01");

  const char* expected = "\"hello \\\"world\\\"\\nnew\\tline\\\\back\\u0001\"";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) == strlen(expected));
  CHECK(Fail, memcmp(buf, expected, jw_length(&jw)) == 0);
  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

static int
test_overflow(void)
{
  char buf[8];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_string(&jw, "this is way too long for the buffer");
  CHECK(Fail, jw_error(&jw));
  return 0;

Fail:
  return 1;
}

static int
test_array_commas(void)
{
  char buf[128];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_array_begin(&jw);
  jw_string(&jw, "a");
  jw_string(&jw, "b");
  jw_string(&jw, "c");
  jw_array_end(&jw);

  const char* expected = "[\"a\",\"b\",\"c\"]";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) == strlen(expected));
  CHECK(Fail, memcmp(buf, expected, jw_length(&jw)) == 0);
  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

static int
test_uint(void)
{
  char buf[64];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_uint(&jw, 18446744073709551615ULL);

  const char* expected = "18446744073709551615";
  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) == strlen(expected));
  CHECK(Fail, memcmp(buf, expected, jw_length(&jw)) == 0);
  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

static int
test_zarr_metadata(void)
{
  char buf[2048];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "array");
  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  jw_uint(&jw, 100);
  jw_uint(&jw, 200);
  jw_uint(&jw, 300);
  jw_array_end(&jw);
  jw_key(&jw, "data_type");
  jw_string(&jw, "uint16");
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  jw_uint(&jw, 10);
  jw_uint(&jw, 20);
  jw_uint(&jw, 30);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_key(&jw, "fill_value");
  jw_int(&jw, 0);
  jw_object_end(&jw);

  CHECK(Fail, !jw_error(&jw));
  CHECK(Fail, jw_length(&jw) > 0);

  // Verify it starts and ends correctly
  CHECK(Fail, buf[0] == '{');
  CHECK(Fail, buf[jw_length(&jw) - 1] == '}');

  // Spot-check some substrings
  buf[jw_length(&jw)] = '\0';
  CHECK(Fail, strstr(buf, "\"zarr_format\":3"));
  CHECK(Fail, strstr(buf, "\"node_type\":\"array\""));
  CHECK(Fail, strstr(buf, "\"shape\":[100,200,300]"));
  CHECK(Fail, strstr(buf, "\"chunk_shape\":[10,20,30]"));

  return 0;

Fail:
  log_error("  got: %.*s", (int)jw_length(&jw), buf);
  return 1;
}

int
main(void)
{
  int rc = 0;
  struct
  {
    const char* name;
    int (*fn)(void);
  } tests[] = {
    { "simple_object", test_simple_object },     { "nested", test_nested },
    { "string_escaping", test_string_escaping }, { "overflow", test_overflow },
    { "array_commas", test_array_commas },       { "uint", test_uint },
    { "zarr_metadata", test_zarr_metadata },
  };
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
    int r = tests[i].fn();
    if (r) {
      log_error("  FAIL: %s", tests[i].name);
      rc = 1;
    } else {
      log_info("  PASS: %s", tests[i].name);
    }
  }
  return rc;
}
