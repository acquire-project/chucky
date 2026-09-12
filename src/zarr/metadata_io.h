// Private Zarr metadata submission over an injected store and optional pool.
#pragma once

struct shard_pool;
struct store;
struct strbuf;

// Submit prefix/zarr.json (or zarr.json for an empty/NULL prefix). A pool with
// queue_metadata copies the key and JSON before returning; other stores write
// synchronously. Success means the snapshot was accepted, so its buffers may
// be released immediately. A later queued IO failure is sticky on the pool.
int
zarr_metadata_submit(struct store* store,
                     struct shard_pool* pool,
                     const char* prefix,
                     const struct strbuf* json);

// Complete accepted queued metadata and report pool errors. This is a no-op
// without queue_metadata: ordinary S3 metadata must not flush shard uploads.
// Synchronous callers wait after submission; completion errors do not undo
// accepted snapshots and remain sticky, just as for asynchronous publication.
int
zarr_metadata_wait(struct shard_pool* pool);
