#include "util/prelude.h"
#include "zarr/filesystem_write.h"

static int
check_split(uint64_t nbytes, size_t alignment, uint64_t expected_count)
{
  const struct filesystem_write write = {
    .offset = 3 * (uint64_t)alignment,
    .nbytes = nbytes,
    .alignment = alignment,
  };
  const uint64_t count = filesystem_write_count(&write);
  CHECK(Fail, count == expected_count);

  uint64_t covered = 0;
  uint64_t smallest = UINT64_MAX;
  uint64_t largest = 0;
  for (uint64_t i = 0; i < count; ++i) {
    struct filesystem_write_part part;
    CHECK(Fail, filesystem_write_at(&write, i, &part) == 0);
    CHECK(Fail, part.offset == write.offset + covered);
    CHECK(Fail, part.nbytes % alignment == 0);
    CHECK(Fail, part.offset % alignment == 0);
    CHECK(Fail, part.nbytes <= FILESYSTEM_WRITE_MAX_BYTES);
    if (part.nbytes < smallest)
      smallest = part.nbytes;
    if (part.nbytes > largest)
      largest = part.nbytes;
    covered += part.nbytes;
  }
  CHECK(Fail, covered == nbytes);
  CHECK(Fail, largest - smallest <= alignment);
  return 0;

Fail:
  return 1;
}

static int
test_splits(void)
{
  CHECK(Fail, check_split(FILESYSTEM_WRITE_MAX_BYTES + 4096, 4096, 2) == 0);
  CHECK(Fail,
        check_split(
          2 * FILESYSTEM_WRITE_MAX_BYTES + 4 * 1024 * 1024, 4096, 3) == 0);
  return 0;

Fail:
  return 1;
}

static int
test_single_write(void)
{
  const struct filesystem_write write = {
    .offset = 7,
    .nbytes = FILESYSTEM_WRITE_MAX_BYTES,
  };
  struct filesystem_write_part part;
  CHECK(Fail, filesystem_write_count(&write) == 1);
  CHECK(Fail, filesystem_write_at(&write, 0, &part) == 0);
  CHECK(Fail, part.offset == write.offset);
  CHECK(Fail, part.nbytes == write.nbytes);
  const struct filesystem_write empty = { 0 };
  CHECK(Fail, filesystem_write_count(&empty) == 0);
  CHECK(Fail, filesystem_write_at(&empty, 0, &part) != 0);
  return 0;

Fail:
  return 1;
}

static int
test_invalid_split(void)
{
  const struct filesystem_write write = {
    .offset = 1,
    .nbytes = FILESYSTEM_WRITE_MAX_BYTES + 4096,
    .alignment = 4096,
  };
  struct filesystem_write_part part;
  CHECK(Fail, filesystem_write_count(&write) == 2);
  CHECK(Fail, filesystem_write_at(&write, 0, &part) != 0);
  CHECK(Fail, filesystem_write_count(NULL) == 0);
  CHECK(Fail, filesystem_write_at(NULL, 0, &part) != 0);
  CHECK(Fail, filesystem_write_at(&write, 2, &part) != 0);
  return 0;

Fail:
  return 1;
}

int
main(void)
{
  return test_splits() || test_single_write() || test_invalid_split();
}
