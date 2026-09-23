#include "platform/topology.h"
#include "test_platform.h"
#include "util/prelude.h"

#include <string.h>

static int
test_cpu_lists(const char* path)
{
  FILE* file = NULL;
  unsigned char cpus[1024], expected[1024] = { 0 };
  const struct
  {
    const char* text;
    enum platform_cpu_list_result result;
  } cases[] = {
    { "", PLATFORM_CPU_LIST_INVALID },
    { "\n", PLATFORM_CPU_LIST_INVALID },
    { "-1", PLATFORM_CPU_LIST_INVALID },
    { "3-1", PLATFORM_CPU_LIST_INVALID },
    { "1-", PLATFORM_CPU_LIST_INVALID },
    { "1,", PLATFORM_CPU_LIST_INVALID },
    { "1x", PLATFORM_CPU_LIST_INVALID },
    { "1\n2", PLATFORM_CPU_LIST_INVALID },
    { "1024", PLATFORM_CPU_LIST_TOO_LARGE },
    { "0-1024", PLATFORM_CPU_LIST_TOO_LARGE },
    { "999999999999999999999999", PLATFORM_CPU_LIST_TOO_LARGE },
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    file = fopen(path, "w+b");
    CHECK(Fail, file);
    CHECK(Fail, fputs(cases[i].text, file) >= 0);
    rewind(file);
    CHECK(Fail,
          platform_parse_cpu_list(file, cpus, sizeof(cpus)) == cases[i].result);
    fclose(file);
    file = NULL;
  }

  file = fopen(path, "w+b");
  CHECK(Fail, file);
  CHECK(Fail, fputs("0-2,8,1023\n", file) >= 0);
  rewind(file);
  CHECK(Fail,
        platform_parse_cpu_list(file, cpus, sizeof(cpus)) ==
          PLATFORM_CPU_LIST_OK);
  expected[0] = expected[1] = expected[2] = expected[8] = expected[1023] = 1;
  CHECK(Fail, memcmp(cpus, expected, sizeof(cpus)) == 0);
  fclose(file);

  file = fopen(path, "w+b");
  CHECK(Fail, file);
  CHECK(Fail, fputs("1023", file) >= 0);
  rewind(file);
  CHECK(Fail,
        platform_parse_cpu_list(file, cpus, sizeof(cpus)) ==
          PLATFORM_CPU_LIST_OK);
  memset(expected, 0, sizeof(expected));
  expected[1023] = 1;
  CHECK(Fail, memcmp(cpus, expected, sizeof(cpus)) == 0);
  fclose(file);
  return 0;
Fail:
  if (file)
    fclose(file);
  return 1;
}

int
main(void)
{
  char directory[1024], path[1100];
  if (test_tmpdir_create(directory, sizeof(directory)))
    return 1;
  snprintf(path, sizeof(path), "%s/cpulist", directory);
  const int failed = test_cpu_lists(path);
  test_tmpdir_remove(directory);
  return failed;
}
