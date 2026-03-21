# S3 Storage Guide

## How Shards Map to S3

Each Zarr v3 shard becomes a single S3 object. The [AWS Common Runtime
(CRT)][aws-crt] uses [multipart upload][mpu] automatically, coalescing
compressed chunks into parts of `part_size` bytes and uploading them
concurrently. Small objects (e.g. `zarr.json` metadata) use a simple PUT.

S3 allows at most **10,000 parts** per [multipart upload][s3-limits].
The sink rejects configurations that could exceed this limit (see
[Choosing `part_size`](#choosing-part_size)).

## Configuration

The S3 transport is configured via `zarr_s3_config`, defined in
[`zarr_s3_sink.h`](../src/zarr/zarr_s3_sink.h). The transport-specific
fields are:

| Field | Default | Description |
|-------|---------|-------------|
| `region` | *required* | AWS region (e.g. `"us-east-1"`) |
| `endpoint` | *required* | S3-compatible endpoint URL (e.g. `"https://s3.us-east-1.amazonaws.com"` or `"http://localhost:9000"`) |
| `part_size` | 8 MiB | [Multipart upload][mpu] part size (see below) |
| `throughput_gbps` | 10.0 | Target throughput in gigabits/s for the CRT |
| `max_retries` | 10 | Retry count per part (0 = CRT default) |
| `backoff_scale_ms` | 500 | Exponential backoff scale in ms (0 = CRT default) |
| `max_backoff_secs` | 20 | Maximum backoff delay in seconds (0 = CRT default) |
| `timeout_ns` | 0 | Timeout per upload wait (0 = infinite) |

### Defaults and validation

Call `zarr_s3_config_set_defaults()` to fill zero-valued optional fields
with their defaults, or let `zarr_s3_sink_create()` do it for you — it
calls `set_defaults` followed by `zarr_s3_config_validate()`
internally. You can also call `validate` yourself for early error
reporting.

```c
zarr_s3_config_set_defaults(&cfg);       // fill part_size, throughput_gbps
if (zarr_s3_config_validate(&cfg))       // check required fields + part count
  return error;
struct zarr_s3_sink* sink = zarr_s3_sink_create(&cfg);

// Connect to the streaming pipeline:
struct shard_sink* ss = zarr_s3_sink_as_shard_sink(sink);

// ... stream data ...

// Drain pending uploads without destroying the sink:
zarr_s3_sink_flush(sink);

// Tear down (also drains pending uploads):
zarr_s3_sink_destroy(sink);
```

Multiscale equivalents: `zarr_s3_multiscale_config_set_defaults()`,
`zarr_s3_multiscale_config_validate()`, `zarr_s3_multiscale_sink_flush()`.

### Credentials

The CRT uses the [standard credential provider chain][cred-chain]:
environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`),
`~/.aws/credentials`, or an IAM instance role.

### Choosing `part_size`

The default 8 MiB works well for most workloads (~80 GB max object
size). Reasons to change it:

- **Large shards** — if a shard exceeds ~80 GB, increase `part_size`
  (up to 5 GiB) or reduce `chunks_per_shard` to stay under the
  10,000-part limit.
- **High-latency links** — larger parts mean fewer round-trips but each
  failed part is more expensive to retry.
- **Low memory** — the CRT buffers a few parts in flight; smaller parts
  reduce peak memory.

## Bucket Lifecycle Policy

If a process crashes or is killed during a multipart upload, S3 retains
the incomplete upload indefinitely. These orphaned uploads consume
storage and incur costs.

> **Recommendation:** Configure an
> [`AbortIncompleteMultipartUpload`][lifecycle] lifecycle rule on any
> bucket that receives shard uploads.

Save the following as `lifecycle.json`:

```json
{
  "Rules": [
    {
      "ID": "abort-incomplete-uploads",
      "Status": "Enabled",
      "Filter": {},
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 1
      }
    }
  ]
}
```

Apply with the [AWS CLI][put-lifecycle]:

```sh
aws s3api put-bucket-lifecycle-configuration \
  --bucket YOUR_BUCKET \
  --lifecycle-configuration file://lifecycle.json
```

## Error Handling

The AWS CRT retries failed part uploads automatically using [exponential
backoff with jitter][retry]. Errors that reach the sink are therefore persistent
failures — the CRT's retry budget has been exhausted.

When a part upload fails after retries:

1. The sink sets a sticky error flag.
2. The upload is **aborted** — the CRT sends an
   [`AbortMultipartUpload`][abort-mpu] request to clean up server-side
   state. The shard object is not created.
3. The error propagates to the caller via `zarr_s3_sink_has_error()` or
   the return value of `zarr_s3_sink_destroy()`.

On normal shutdown, `zarr_s3_sink_destroy()` waits for all finalized
uploads to complete. Any upload still in progress (not yet finalized) is
aborted.

> **Recommendation:** Even with abort-on-error, keep the lifecycle rule
> above as a safety net for hard crashes.

## Testing with MinIO

Set the `endpoint` field to point at a local [MinIO][minio] instance.
Path-style addressing and plaintext HTTP are used automatically for
`http://` endpoints.

### First-time setup

```sh
docker run -d \
  --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio server /data --console-address ":9001"
```

Note: this uses an anonymous Docker volume for `/data`. Data persists
across `docker stop`/`docker start` but is lost on `docker rm`. For
durable storage, add `-v /path/on/host:/data`.

Create a test bucket via the MinIO console at `http://localhost:9001`
(log in with `minioadmin` / `minioadmin`), or with the AWS CLI:

```sh
aws --endpoint-url http://localhost:9000 s3 mb s3://test-bucket
```

### Subsequent runs

```sh
docker start minio
```

### Configuration for tests

```c
struct zarr_s3_config cfg = {
  .bucket   = "test-bucket",
  .endpoint = "http://localhost:9000",
  .region   = "us-east-1",
  // ...
};
```

Set `AWS_ACCESS_KEY_ID=minioadmin` and
`AWS_SECRET_ACCESS_KEY=minioadmin` in your environment so the CRT
credential chain picks them up.

## End-to-End Example

A minimal example that streams a 3D array (with a streaming time
dimension) to S3 using the GPU pipeline:

```c
#include "zarr_s3_sink.h"
#include "gpu/stream.h"
#include "dimension.h"
#include "writer.h"

// 1. Define dimensions: t (streaming) × y × x
uint64_t sizes[] = { 0, 256, 256 };
struct dimension dims[3];
dims_create(dims, "tyx", sizes);

// 2. Choose chunk sizes within a GPU memory budget
struct tile_stream_configuration stream_cfg = {
  .dtype      = dtype_u16,
  .rank       = 3,
  .dimensions = dims,
  .codec      = CODEC_ZSTD,
};

uint8_t ratios[] = { 0, 1, 1 };  // don't chunk along t, equal y:x
tile_stream_gpu_advise_chunk_sizes(
  &stream_cfg, 128 * 1024, ratios, 2ULL << 30);  // 128 KiB target, 2 GB budget

// 3. Create the S3 sink (uses the chunk sizes chosen above)
struct zarr_s3_config s3cfg = {
  .bucket     = "my-bucket",
  .prefix     = "experiment-001",
  .array_name = "0",
  .region     = "us-east-1",
  .endpoint   = "http://localhost:9000",
  .data_type  = dtype_u16,
  .rank       = 3,
  .dimensions = dims,
  .codec      = CODEC_ZSTD,
};

struct zarr_s3_sink* sink = zarr_s3_sink_create(&s3cfg);

// 4. Create the streaming pipeline
stream_cfg.shard_sink = zarr_s3_sink_as_shard_sink(sink);
struct tile_stream_gpu* stream = tile_stream_gpu_create(&stream_cfg);
struct writer* w = tile_stream_gpu_writer(stream);

// 5. Stream frames
for (int t = 0; t < nframes; ++t) {
  uint16_t* frame = acquire_frame();
  struct slice sl = { frame, (char*)frame + 256 * 256 * sizeof(uint16_t) };
  writer_append(w, sl);
}
writer_flush(w);

// 6. Tear down
tile_stream_gpu_destroy(stream);
zarr_s3_sink_destroy(sink);
```

<!-- references -->
[mpu]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html
[s3-limits]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
[lifecycle]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpu-abort-incomplete-mpu-lifecycle-config.html
[put-lifecycle]: https://docs.aws.amazon.com/cli/latest/reference/s3api/put-bucket-lifecycle-configuration.html
[retry]: https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html
[abort-mpu]: https://docs.aws.amazon.com/AmazonS3/latest/API/API_AbortMultipartUpload.html
[minio]: https://min.io/docs/minio/container/index.html
[cred-chain]: https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html
[aws-crt]: https://docs.aws.amazon.com/sdkref/latest/guide/common-runtime.html
