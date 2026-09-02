#include "util/prelude.h"
#include "zarr/filesystem_command_map.h"

#include <stdlib.h>

static int
check_split(uint64_t nbytes, size_t alignment, uint64_t expected_count)
{
  void* data = malloc((size_t)nbytes);
  CHECK(Fail, data);
  const struct filesystem_command command = {
    .kind = FILESYSTEM_COMMAND_DATA,
    .source = data,
    .file_offset = 3 * (uint64_t)alignment,
    .nbytes = nbytes,
    .alignment = alignment,
  };
  const uint64_t count = filesystem_command_count(&command);
  CHECK(Fail, count == expected_count);

  uint64_t covered = 0;
  uint64_t smallest = UINT64_MAX;
  uint64_t largest = 0;
  for (uint64_t i = 0; i < count; ++i) {
    struct filesystem_command part;
    CHECK(Fail, filesystem_command_at(&command, i, &part) == 0);
    CHECK(Fail, part.kind == FILESYSTEM_COMMAND_DATA);
    CHECK(Fail, part.source == (const char*)data + covered);
    CHECK(Fail, part.file_offset == command.file_offset + covered);
    CHECK(Fail, part.nbytes % alignment == 0);
    CHECK(Fail, part.file_offset % alignment == 0);
    CHECK(Fail, part.nbytes >= FILESYSTEM_COMMAND_MIN_BYTES);
    CHECK(Fail, part.nbytes <= FILESYSTEM_COMMAND_MAX_BYTES);
    if (part.nbytes < smallest)
      smallest = part.nbytes;
    if (part.nbytes > largest)
      largest = part.nbytes;
    covered += part.nbytes;
  }
  CHECK(Fail, covered == nbytes);
  CHECK(Fail, largest - smallest <= alignment);
  free(data);
  return 0;

Fail:
  free(data);
  return 1;
}

static int
test_bounds_and_remainders(void)
{
  CHECK(Fail, check_split(FILESYSTEM_COMMAND_MAX_BYTES + 4096, 4096, 2) == 0);
  CHECK(
    Fail,
    check_split(2 * FILESYSTEM_COMMAND_MAX_BYTES + FILESYSTEM_COMMAND_MIN_BYTES,
                4096,
                3) == 0);
  return 0;

Fail:
  return 1;
}

static int
test_unchanged_commands(void)
{
  const char byte = 0;
  const struct filesystem_command cases[] = {
    { .kind = FILESYSTEM_COMMAND_DATA,
      .source = &byte,
      .file_offset = 7,
      .nbytes = 0 },
    { .kind = FILESYSTEM_COMMAND_DATA,
      .source = &byte,
      .file_offset = 7,
      .nbytes = FILESYSTEM_COMMAND_MAX_BYTES },
    { .kind = FILESYSTEM_COMMAND_UNCHANGED,
      .source = &byte,
      .file_offset = 7,
      .nbytes = 3 * FILESYSTEM_COMMAND_MAX_BYTES },
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    struct filesystem_command part;
    CHECK(Fail, filesystem_command_count(&cases[i]) == 1);
    CHECK(Fail, filesystem_command_at(&cases[i], 0, &part) == 0);
    CHECK(Fail, part.kind == cases[i].kind);
    CHECK(Fail, part.source == cases[i].source);
    CHECK(Fail, part.file_offset == cases[i].file_offset);
    CHECK(Fail, part.nbytes == cases[i].nbytes);
  }
  return 0;

Fail:
  return 1;
}

static int
test_invalid_alignment(void)
{
  const struct filesystem_command command = {
    .kind = FILESYSTEM_COMMAND_DATA,
    .source = (const void*)1,
    .file_offset = 1,
    .nbytes = FILESYSTEM_COMMAND_MAX_BYTES + 4096,
    .alignment = 4096,
  };
  struct filesystem_command part;
  CHECK(Fail, filesystem_command_count(&command) == 2);
  CHECK(Fail, filesystem_command_at(&command, 0, &part) != 0);
  CHECK(Fail, filesystem_command_count(NULL) == 0);
  CHECK(Fail, filesystem_command_at(NULL, 0, &part) != 0);
  CHECK(Fail, filesystem_command_at(&command, 2, &part) != 0);
  return 0;

Fail:
  return 1;
}

int
main(void)
{
  if (test_bounds_and_remainders())
    return 1;
  if (test_unchanged_commands())
    return 1;
  if (test_invalid_alignment())
    return 1;
  return 0;
}
