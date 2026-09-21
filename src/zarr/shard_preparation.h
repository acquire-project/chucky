// Private filesystem preparation worker and ownership of unused resources.
// Callers serialize posting and adoption; cancel only after delivery stops.
#pragma once

#include "zarr/io_backend.h"

#include <stdatomic.h>
#include <stdint.h>

struct io_backend_fs;
struct shard_preparation;

struct shard_preparation*
shard_preparation_create(uint64_t nslots,
                         int open_flags,
                         _Atomic int* io_error,
                         void* wrap_ctx,
                         struct io_backend (*wrap)(void*, struct io_backend));

int
shard_preparation_post(struct shard_preparation* p,
                       uint64_t slot,
                       const char* path,
                       uint64_t capacity);

// Waiting and cancellation accept NULL when preparation has never started.
int
shard_preparation_wait(struct shard_preparation* p, uint64_t slot);

// 1 transfers a prepared file to backend, 0 means no preparation, -1 fails.
// Failure leaves ownership here so cancellation can reclaim the file.
int
shard_preparation_adopt(struct shard_preparation* p,
                        uint64_t slot,
                        const char* path,
                        struct io_backend_fs* backend,
                        struct io_file_token* token,
                        uint64_t* capacity);

int
shard_preparation_cancel(struct shard_preparation* p,
                         uint64_t first,
                         uint64_t count);

// Cancels unused files and joins the worker. NULL is harmless.
void
shard_preparation_destroy(struct shard_preparation* p);
