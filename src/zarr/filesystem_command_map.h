#pragma once

#include <stddef.h>
#include <stdint.h>

#define FILESYSTEM_COMMAND_MIN_BYTES (4ull << 20)
#define FILESYSTEM_COMMAND_MAX_BYTES (64ull << 20)

enum filesystem_command_kind
{
  FILESYSTEM_COMMAND_DATA = 0,
  FILESYSTEM_COMMAND_UNCHANGED,
};

struct filesystem_command
{
  enum filesystem_command_kind kind;
  const void* source;
  uint64_t file_offset;
  uint64_t nbytes;
  size_t alignment;
};

uint64_t
filesystem_command_count(const struct filesystem_command* command);

int
filesystem_command_at(const struct filesystem_command* command,
                      uint64_t index,
                      struct filesystem_command* part);
