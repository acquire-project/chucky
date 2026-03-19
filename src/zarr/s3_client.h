#pragma once

#include <stddef.h>

// Thin wrapper around aws-c-s3. Manages CRT lifecycle, credential chain,
// signing, and provides blocking put/upload operations.

struct s3_client;

struct s3_client_config
{
  const char* region;   // NULL = from env/config
  const char* endpoint; // NULL = default AWS, e.g. "http://localhost:9000"
};

struct s3_client*
s3_client_create(const struct s3_client_config* cfg);

void
s3_client_destroy(struct s3_client* c);

// Blocking PUT of a small object (zarr.json metadata).
// Returns 0 on success, non-zero on error.
int
s3_client_put(struct s3_client* c,
              const char* bucket,
              const char* key,
              const void* data,
              size_t len);

// --- Streaming upload for large objects (shards) ---

struct s3_upload;

// Begin a streaming upload. The CRT will use multipart upload automatically
// for large objects. Returns NULL on error.
struct s3_upload*
s3_upload_begin(struct s3_client* c, const char* bucket, const char* key);

// Feed data into the upload. Blocks until the CRT has consumed the data
// (fast — just copies into internal buffer). Returns 0 on success.
int
s3_upload_write(struct s3_upload* u, const void* data, size_t len);

// Signal EOF and block until the upload is fully complete (all parts uploaded,
// CompleteMultipartUpload finished). Returns 0 on success.
int
s3_upload_finish(struct s3_upload* u);

// Cancel an in-progress upload and free resources.
void
s3_upload_abort(struct s3_upload* u);
