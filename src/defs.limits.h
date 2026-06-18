#pragma once

// Compile-time limits — collected in one place for easy tuning.

#define MAX_RANK 64
#define HALF_MAX_RANK (MAX_RANK / 2)
#define LOD_MAX_NDIM HALF_MAX_RANK
#define LOD_MAX_LEVELS 32

// Generations of per-fc LOD timing events. The delivery worker reads a
// drained batch's timing while the producer re-records the next same-fc
// batch's events; at the worst case three batches are live at once (one
// draining, one kicked-and-pending, one filling), so each owns its own
// generation for its whole lifetime (#154).
#define LOD_TIMING_SLOTS 3
#define MAX_ZARR_RANK (HALF_MAX_RANK)

// S3
#define S3_MAX_PARTS 10000
#define S3_DEFAULT_PART_SIZE (8 * 1024 * 1024)
#define S3_DEFAULT_THROUGHPUT_GBPS 10.0

// Shard backend limits — applied uniformly across sinks (conservative).
// One chunk per upload part, so parts-count = chunks per shard.
#define MAX_PARTS_PER_SHARD S3_MAX_PARTS
#define MAX_BYTES_PER_PART (5ull * 1024 * 1024 * 1024) // S3 single-part ceiling
