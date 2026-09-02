#include "zarr/filesystem_command_map.h"

#include <stdint.h>

uint64_t
filesystem_command_count(const struct filesystem_command* command)
{
  if (!command || command->kind != FILESYSTEM_COMMAND_DATA ||
      command->nbytes <= FILESYSTEM_COMMAND_MAX_BYTES)
    return command ? 1 : 0;
  return 1 + (command->nbytes - 1) / FILESYSTEM_COMMAND_MAX_BYTES;
}

int
filesystem_command_at(const struct filesystem_command* command,
                      uint64_t index,
                      struct filesystem_command* part)
{
  if (!command || !part)
    return 1;
  const uint64_t count = filesystem_command_count(command);
  if (index >= count)
    return 1;
  if (count == 1) {
    *part = *command;
    return 0;
  }

  const uint64_t alignment = command->alignment ? command->alignment : 1;
  if (!command->source || alignment > FILESYSTEM_COMMAND_MAX_BYTES ||
      FILESYSTEM_COMMAND_MAX_BYTES % alignment != 0 ||
      command->file_offset % alignment != 0 || command->nbytes % alignment != 0)
    return 1;

  const uint64_t units = command->nbytes / alignment;
  const uint64_t units_per_part = units / count;
  const uint64_t larger_parts = units % count;
  const uint64_t part_units = units_per_part + (index < larger_parts ? 1 : 0);
  const uint64_t units_before =
    index * units_per_part + (index < larger_parts ? index : larger_parts);
  const uint64_t byte_offset = units_before * alignment;
  const uint64_t part_bytes = part_units * alignment;
  if (part_bytes < FILESYSTEM_COMMAND_MIN_BYTES ||
      part_bytes > FILESYSTEM_COMMAND_MAX_BYTES ||
      byte_offset > command->nbytes ||
      part_bytes > command->nbytes - byte_offset ||
      command->file_offset > UINT64_MAX - byte_offset)
    return 1;

  *part = *command;
  part->source = (const uint8_t*)command->source + byte_offset;
  part->file_offset = command->file_offset + byte_offset;
  part->nbytes = part_bytes;
  return 0;
}
