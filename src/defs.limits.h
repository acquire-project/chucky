#pragma once

// Compile-time limits — collected in one place for easy tuning.

#define MAX_RANK 64
#define HALF_MAX_RANK (MAX_RANK / 2)
#define LOD_MAX_NDIM HALF_MAX_RANK
#define LOD_MAX_LEVELS 32

// Worst case three batches are live at once (draining, kicked-and-pending,
// filling), so each needs its own timing buffer (#154).
// One measurement per epoch, read by the producer a few epochs later once the
// device has caught up. Enough slack for the epochs in flight, not one per
// batch — a per-batch slot is overwritten by every epoch after the first.
#define LOD_TIMING_SLOTS 8

// Buckets spanning 1us to 100s, twenty per decade, so a reported time is within
// about 12% of the truth.
#define APPEND_LATENCY_BUCKETS 160
#define APPEND_LATENCY_MIN_MS 0.001
#define APPEND_LATENCY_PER_DECADE 20

// More than the two staging slots, so a scatter measurement can outlive the
// dispatch that produced it and still be read after it completes.
#define SCATTER_TIMING_SLOTS 4
#define MAX_ZARR_RANK (HALF_MAX_RANK)

// S3
#define S3_MAX_PARTS 10000
#define S3_DEFAULT_PART_SIZE (8 * 1024 * 1024)
#define S3_DEFAULT_THROUGHPUT_GBPS 10.0

// Shard backend limits — applied uniformly across sinks (conservative).
// One chunk per upload part, so parts-count = chunks per shard.
#define MAX_PARTS_PER_SHARD S3_MAX_PARTS
#define MAX_BYTES_PER_PART (5ull * 1024 * 1024 * 1024) // S3 single-part ceiling
