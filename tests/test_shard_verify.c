#include "test_shard_verify.h"

#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>
#include <zstd.h>

int
shard_index_parse(const uint8_t* buf,
                  size_t shard_size,
                  size_t chunks_per_shard,
                  uint64_t* offsets,
                  uint64_t* sizes)
{
  size_t index_data_bytes = chunks_per_shard * 2 * sizeof(uint64_t);
  if (shard_size <= index_data_bytes + 4)
    return 1;
  const uint8_t* index_ptr = buf + shard_size - index_data_bytes - 4;
  for (size_t i = 0; i < chunks_per_shard; ++i) {
    memcpy(&offsets[i], index_ptr + i * 16, sizeof(uint64_t));
    memcpy(&sizes[i], index_ptr + i * 16 + 8, sizeof(uint64_t));
  }
  return 0;
}

int
verify_offsets_monotonic(const size_t* offsets, uint64_t n)
{
  CHECK(Fail, offsets[0] == 0);
  for (uint64_t j = 1; j <= n; ++j)
    CHECK(Fail, offsets[j] >= offsets[j - 1]);
  return 0;

Fail:
  return 1;
}

int
chunk_decompress_verify_u16(const uint8_t* comp_data,
                            size_t comp_size,
                            size_t chunk_bytes,
                            uint64_t chunk_stride,
                            uint16_t expected_val)
{
  uint8_t* decomp = (uint8_t*)calloc(1, chunk_bytes);
  if (!decomp)
    return 1;

  size_t result = ZSTD_decompress(decomp, chunk_bytes, comp_data, comp_size);
  if (ZSTD_isError(result) || result != chunk_bytes) {
    free(decomp);
    return 1;
  }

  const uint16_t* got = (const uint16_t*)decomp;
  int errors = 0;
  for (uint64_t e = 0; e < chunk_stride; ++e) {
    if (got[e] != expected_val)
      errors++;
  }
  free(decomp);
  return errors ? 1 : 0;
}

int
chunk_decompress(const uint8_t* comp_data,
                 size_t comp_size,
                 void* out,
                 size_t out_bytes)
{
  size_t result = ZSTD_decompress(out, out_bytes, comp_data, comp_size);
  if (ZSTD_isError(result) || result != out_bytes)
    return 1;
  return 0;
}
