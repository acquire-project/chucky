// Abstract key-value store for metadata I/O.
// Implementations: store_fs (filesystem), store_s3 (AWS S3).
#pragma once

#include <stddef.h>
#include <stdint.h>

struct shard_pool;

struct store
{
  // Write a small blob at key (relative to store root).
  // Used for zarr.json metadata files. Synchronous.
  int (*put)(struct store* self, const char* key, const void* data, size_t len);

  // Ensure key prefix directories exist (no-op for object stores).
  int (*mkdirs)(struct store* self, const char* key);

  // Create a shard writer pool with nslots writer slots.
  // The pool borrows backend resources from the store.
  // Caller owns the returned pool and must destroy it before the store.
  struct shard_pool* (*create_pool)(struct store* self, uint64_t nslots);

  // Coarse overwrite-guard. Returns 1 if zarr.json exists at the store root,
  // 0 if absent, -1 on IO error. O(1) — one stat / HEAD.
  int (*has_existing_data)(struct store* self);

  void (*destroy)(struct store* self);
};
