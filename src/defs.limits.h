#pragma once

// Compile-time limits — collected in one place for easy tuning.

#define MAX_RANK 64
#define HALF_MAX_RANK (MAX_RANK / 2)
#define LOD_MAX_NDIM 64
#define LOD_MAX_LEVELS 32
#define MAX_BATCH_EPOCHS 128
#define MAX_ZARR_RANK (HALF_MAX_RANK)

// Zarr group metadata
#define ZARR_GROUP_JSON_CAP 256

// S3
#define S3_MAX_PARTS 10000
#define S3_DEFAULT_PART_SIZE (8 * 1024 * 1024)
#define S3_DEFAULT_THROUGHPUT_GBPS 10.0
