#ifndef TEST_SHARD_VERIFY_H
#define TEST_SHARD_VERIFY_H

#include <stddef.h>
#include <stdint.h>

// Parse the shard index from the end of a shard buffer.
// Layout: [chunk_data...][offset0 u64][size0 u64]...[offsetN u64][sizeN
// u64][crc32c u32] Returns 0 on success, 1 on failure.
int
shard_index_parse(const uint8_t* buf,
                  size_t shard_size,
                  size_t chunks_per_shard,
                  uint64_t* offsets,
                  uint64_t* sizes);

// Check the trailing index block against its stored checksum. A shard that
// has not been closed out has chunk data where the index belongs, so this is
// how a test tells a finalized shard from one still being written.
// Returns 0 if the checksum matches, 1 otherwise.
int
shard_index_check_crc(const uint8_t* buf,
                      size_t shard_size,
                      size_t chunks_per_shard);

// Verify that offsets are monotonically non-decreasing, starting at 0.
// offsets[0..n] inclusive (n+1 entries).
// Returns 0 on success, 1 on failure.
int
verify_offsets_monotonic(const size_t* offsets, uint64_t n);

// Decompress a ZSTD-compressed chunk and verify all u16 elements equal
// expected_val. Returns 0 on success, 1 on failure.
int
chunk_decompress_verify_u16(const uint8_t* comp_data,
                            size_t comp_size,
                            size_t chunk_bytes,
                            uint64_t chunk_stride,
                            uint16_t expected_val);

// Decompress a ZSTD-compressed chunk into caller-provided buffer.
// Returns 0 on success, 1 on failure.
int
chunk_decompress(const uint8_t* comp_data,
                 size_t comp_size,
                 void* out,
                 size_t out_bytes);

#endif // TEST_SHARD_VERIFY_H
