# S3 Storage Guide

## How Shards Map to S3

Each Zarr v3 shard becomes a single S3 object. The [AWS Common Runtime
(CRT)][aws-crt] uses [multipart upload][mpu] automatically, coalescing
compressed chunks into parts of `part_size` bytes and uploading them
concurrently. Small objects (e.g. `zarr.json` metadata) use a simple PUT.

Unlike a local filesystem, S3 objects are immutable — you cannot seek,
append, or partially overwrite them. A shard must be written as a single
upload from start to finish. If the upload fails or is interrupted the
object is not created (see [Error Handling](#error-handling) and
[Bucket Lifecycle Policy](#bucket-lifecycle-policy)).

S3 also imposes hard limits on multipart uploads: at most **10,000 parts**
and a maximum part size of 5 GiB per [upload][s3-limits]. The sink
rejects configurations that could exceed the part-count limit (see
[Limitations](#limitations)).

## Configuration

The S3 transport is configured via `store_s3_config`, defined in
[`store_s3.h`](../src/zarr/store_s3.h). The transport-specific
fields are:

| Field | Default | Description |
|-------|---------|-------------|
| `region` | *required* | AWS region (e.g. `"us-east-1"`) |
| `endpoint` | *required* | S3-compatible endpoint URL (e.g. `"https://s3.us-east-1.amazonaws.com"` or `"http://localhost:9000"`) |
| `part_size` | 8 MiB | [Multipart upload][mpu] part size (see [Limitations](#limitations)) |
| `throughput_gbps` | 10.0 | Target throughput in gigabits/s for the CRT |
| `max_retries` | 10 | Retry count per part |
| `backoff_scale_ms` | 500 | Exponential backoff scale in ms |
| `max_backoff_secs` | 20 | Maximum backoff delay in seconds |
| `timeout_ns` | 0 (infinite) | Timeout per upload wait |

### Usage

```c
#include "zarr/store_s3.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"
#include "lod/lod_plan.h"

struct store_s3_config scfg = {
  .bucket = "my-bucket",
  .prefix = "data/out.zarr",
  .region = "us-east-1",
  .endpoint = "https://s3.us-east-1.amazonaws.com",
};
store_s3_config_set_defaults(&scfg);  // fill part_size, throughput_gbps

struct store* store = store_s3_create(&scfg);

uint64_t sc[RANK], cps[RANK];
uint64_t sic = dims_compute_shard_geometry(dims, rank, sc, cps);
struct shard_pool* pool = store->create_pool(store, sic);

zarr_write_group(store, "zarr.json", NULL);  // root group
struct zarr_array_config acfg = { ... , .shard_counts = sc,
    .chunks_per_shard = cps, .shard_inner_count = sic };
struct zarr_array* a = zarr_array_create(store, pool, "0", &acfg);

struct shard_sink* ss = zarr_array_as_shard_sink(a);
// ... stream data ...

pool->flush(pool);         // drain pending uploads
zarr_array_destroy(a);
pool->destroy(pool);       // must destroy before store
store->destroy(store);
```

See `docs/formats.md` for multiscale and HCS examples.

### Credentials

The CRT uses the [standard credential provider chain][cred-chain]:
environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`),
`~/.aws/credentials`, or an IAM instance role.

### Limitations

S3 multipart uploads are subject to two hard limits:

- **10,000 parts per upload.** With the default 8 MiB `part_size` this
  caps a single shard at ~80 GB. The sink rejects configurations that
  could exceed this limit at creation time.
- **5 GiB maximum part size.** This sets an absolute ceiling of ~50 TB
  per shard.

If your shards are larger than `part_size × 10,000`, either increase
`part_size` or reduce `chunks_per_shard`.

Other considerations when tuning `part_size`:

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
3. The error propagates to the caller via `pool->has_error(pool)` or
   the return value of `pool->flush(pool)`.

On normal shutdown, `pool->destroy(pool)` waits for all finalized
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
struct store_s3_config cfg = {
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

The snippet below streams a 3D array (with a streaming time dimension) to S3
using the GPU pipeline:

```c
#include "zarr/store_s3.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"
#include "lod/lod_plan.h"
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

uint8_t ratios[] = { 0, 1, 1 };
tile_stream_gpu_advise_chunk_sizes(
  &stream_cfg, 128 * 1024, ratios, 2ULL << 30);

// 3. Create S3 store + pool + array
struct store_s3_config scfg = {
  .bucket   = "my-bucket",
  .prefix   = "experiment-001",
  .region   = "us-east-1",
  .endpoint = "http://localhost:9000",
};
store_s3_config_set_defaults(&scfg);
struct store* store = store_s3_create(&scfg);

uint64_t sc[3], cps[3];
uint64_t sic = dims_compute_shard_geometry(dims, 3, sc, cps);
struct shard_pool* pool = store->create_pool(store, sic);

zarr_write_group(store, "zarr.json", NULL);
struct zarr_array_config acfg = {
  .data_type = dtype_u16, .rank = 3, .dimensions = dims,
  .codec = { .id = CODEC_ZSTD },
  .shard_counts = sc, .chunks_per_shard = cps, .shard_inner_count = sic,
};
struct zarr_array* arr = zarr_array_create(store, pool, "0", &acfg);

// 4. Create the streaming pipeline
struct tile_stream_gpu* stream =
  tile_stream_gpu_create(&stream_cfg, zarr_array_as_shard_sink(arr));
struct writer* w = tile_stream_gpu_writer(stream);

// 5. Stream frames
for (int t = 0; t < nframes; ++t) {
  uint16_t* frame = acquire_frame();
  struct slice sl = { frame, (char*)frame + 256 * 256 * sizeof(uint16_t) };
  writer_append(w, sl);
}
writer_flush(w);

// 6. Tear down (pool before store)
tile_stream_gpu_destroy(stream);
zarr_array_destroy(arr);
pool->destroy(pool);
store->destroy(store);
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
