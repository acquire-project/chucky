#pragma once

#include <stddef.h>
#include <stdint.h>

// Thin wrapper around aws-c-s3. Manages CRT lifecycle, credential chain,
// signing, and provides blocking put/upload operations.

struct s3_client;

struct s3_client_config
{
  const char* region;        // required (e.g. "us-east-1")
  const char* endpoint;      // required (e.g. "https://s3.us-east-1.amazonaws.com")
  size_t part_size;          // 0 = default (8 MiB)
  double throughput_gbps;    // gigabits/s, 0 = default (10.0)
  size_t max_retries;        // 0 = CRT default (10)
  uint32_t backoff_scale_ms; // 0 = CRT default (500)
  uint32_t max_backoff_secs; // 0 = CRT default (20)
  uint64_t timeout_ns;       // 0 = no timeout (infinite)
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

// Signal EOF. The upload completes asynchronously in the CRT background.
// Returns 0 on success (EOF accepted), non-zero on error.
int
s3_upload_finish_async(struct s3_upload* u);

// Block until the upload is fully complete. Returns 0 on success.
// Must be called after finish_async and before destroy.
int
s3_upload_wait(struct s3_upload* u);

// Free the upload struct. Must be called after wait or abort.
void
s3_upload_destroy(struct s3_upload* u);

// Cancel an in-progress upload, wait for cancellation, and free resources.
void
s3_upload_abort(struct s3_upload* u);
