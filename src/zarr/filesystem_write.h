#pragma once

#include <stddef.h>
#include <stdint.h>

#define FILESYSTEM_WRITE_MAX_BYTES (64ull << 20)

struct filesystem_write
{
  uint64_t offset;
  uint64_t nbytes;
  size_t alignment;
};

struct filesystem_write_part
{
  uint64_t offset;
  uint64_t nbytes;
};

uint64_t
filesystem_write_count(const struct filesystem_write* write);

int
filesystem_write_at(const struct filesystem_write* write,
                    uint64_t index,
                    struct filesystem_write_part* part);
