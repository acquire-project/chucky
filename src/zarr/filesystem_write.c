#include "zarr/filesystem_write.h"

#include <stdint.h>

uint64_t
filesystem_write_count(const struct filesystem_write* write)
{
  if (!write || write->nbytes == 0)
    return 0;
  if (write->nbytes <= FILESYSTEM_WRITE_MAX_BYTES)
    return 1;
  return 1 + (write->nbytes - 1) / FILESYSTEM_WRITE_MAX_BYTES;
}

int
filesystem_write_at(const struct filesystem_write* write,
                    uint64_t index,
                    struct filesystem_write_part* part)
{
  if (!write || !part)
    return 1;
  const uint64_t count = filesystem_write_count(write);
  if (index >= count)
    return 1;
  if (count == 1) {
    *part = (struct filesystem_write_part){
      .offset = write->offset,
      .nbytes = write->nbytes,
    };
    return 0;
  }

  const uint64_t alignment = write->alignment ? write->alignment : 1;
  if (alignment > FILESYSTEM_WRITE_MAX_BYTES ||
      FILESYSTEM_WRITE_MAX_BYTES % alignment != 0 ||
      write->offset % alignment != 0 || write->nbytes % alignment != 0)
    return 1;

  const uint64_t units = write->nbytes / alignment;
  const uint64_t units_per_part = units / count;
  const uint64_t larger_parts = units % count;
  const uint64_t part_units = units_per_part + (index < larger_parts);
  const uint64_t units_before =
    index * units_per_part + (index < larger_parts ? index : larger_parts);
  const uint64_t byte_offset = units_before * alignment;
  const uint64_t part_bytes = part_units * alignment;
  if (byte_offset > write->nbytes || part_bytes > write->nbytes - byte_offset ||
      part_bytes > FILESYSTEM_WRITE_MAX_BYTES ||
      write->offset > UINT64_MAX - byte_offset)
    return 1;

  *part = (struct filesystem_write_part){
    .offset = write->offset + byte_offset,
    .nbytes = part_bytes,
  };
  return 0;
}
