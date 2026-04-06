# Formats guide

Chucky writes zarr v3 sharded arrays with optional OME-NGFF and HCS metadata.
The format layers compose — each builds on the one below it.

## Architecture

```
hcs_plate       (src/hcs/)    HCS plate/well/FOV hierarchy
  |
ngff_multiscale (src/ngff/)   OME-NGFF v0.5 multiscale group
  |
zarr_array      (src/zarr/)   Zarr v3 sharded array
  |
store + shard_pool (src/zarr/)  Storage backend (FS or S3)
```

Each layer uses the one below. `hcs_plate` creates `ngff_multiscale`
instances, which create `zarr_array` instances, which use `store` and
`shard_pool` for I/O.

## Quick start

### Single array on filesystem

```c
#include "zarr/store_fs.h"
#include "zarr/zarr_array.h"
#include "zarr/zarr_group.h"

struct store* store = store_fs_create("/data/out.zarr", 0);
store->mkdirs(store, ".");
struct shard_pool* pool = store->create_pool(store, shard_inner_count);

// Write root group + array
zarr_write_group(store, "zarr.json", NULL);
struct zarr_array* a = zarr_array_create(store, pool, "0", &cfg);

// Use the shard_sink interface with the stream
struct shard_sink* sink = zarr_array_as_shard_sink(a);

// Cleanup (pool must be destroyed before store)
zarr_array_destroy(a);
pool->destroy(pool);
store->destroy(store);
```

### OME-NGFF multiscale

```c
#include "ngff/ngff_multiscale.h"
#include "zarr/store_fs.h"
#include "zarr/zarr_group.h"

struct store* store = store_fs_create("/data/out.zarr", 0);
store->mkdirs(store, ".");
struct shard_pool* pool = store->create_pool(store, shard_inner_count);

zarr_write_group(store, "zarr.json", NULL);  // root group

struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 3,
    .dimensions = dims,     // L0 dimensions
    .axes = axes,           // NGFF axis metadata (unit, scale, type)
};

struct ngff_multiscale* ms =
    ngff_multiscale_create(store, pool, "multiscale", &cfg);
struct shard_sink* sink = ngff_multiscale_as_shard_sink(ms);

// Cleanup
ngff_multiscale_destroy(ms);
pool->destroy(pool);
store->destroy(store);
```

### HCS plate

```c
#include "hcs/hcs.h"
#include "zarr/store_fs.h"

struct store* store = store_fs_create("/data/plate.zarr", 0);
store->mkdirs(store, ".");
struct shard_pool* pool = store->create_pool(store, shard_inner_count);

struct hcs_plate_config cfg = {
    .name = "plate",
    .rows = 8,
    .cols = 12,
    .field_count = 1,
    .fov = {
        .data_type = dtype_u16,
        .rank = 3,
        .dimensions = dims,
        .axes = axes,
    },
};

struct hcs_plate* plate = hcs_plate_create(store, pool, &cfg);

// Get shard_sink for well A/1, field 0
struct shard_sink* sink = hcs_plate_fov_sink(plate, 0, 0, 0);

// Cleanup
hcs_plate_destroy(plate);
pool->destroy(pool);
store->destroy(store);
```

### S3 backend

Replace `store_fs_create` with `store_s3_create` — everything else is
identical:

```c
#include "zarr/store_s3.h"

struct store_s3_config s3cfg = {
    .bucket = "my-bucket",
    .prefix = "data/out.zarr",
    .region = "us-east-1",
    .endpoint = "https://s3.us-east-1.amazonaws.com",
};
struct store* store = store_s3_create(&s3cfg);
```

## Key concepts

### Store

`struct store` (`zarr/store.h`) is the key-value I/O abstraction:
- `put(key, data, len)` — write a small blob (metadata files)
- `mkdirs(key)` — ensure directories exist (no-op for S3)
- `create_pool(nslots)` — create a shard writer pool

Implementations: `store_fs` (filesystem), `store_s3` (AWS S3).

### Shard pool

`struct shard_pool` (`zarr/shard_pool.h`) manages reusable writer slots
for streaming shard data:
- `open(slot, key)` — open a writer at the given key
- `record_fence()` / `wait_fence(ev)` — backpressure
- `flush()` / `has_error()` / `pending_bytes()`

The pool is created by the store and must be destroyed before the store.
Multiple format layers (e.g. all levels in a multiscale) can share one pool.

### Geometry

`zarr_array` does not compute shard geometry. The caller provides
pre-computed `shard_counts`, `chunks_per_shard`, and `shard_inner_count`
in the config. Use `dims_compute_shard_geometry()` from `lod/lod_plan.h`
to compute these from a dimension array.

`ngff_multiscale` computes per-level geometry internally via `lod_plan`.
`hcs_plate` delegates to `ngff_multiscale`.

### NGFF axis metadata

`struct ngff_axis` (`ngff/ngff_axis.h`) describes per-dimension metadata:
- `unit` — e.g. `"micrometer"`, `"second"` (NULL omits the field)
- `scale` — physical pixel scale (0 treated as 1.0)
- `type` — `ngff_axis_space`, `ngff_axis_time`, or `ngff_axis_channel`

This is separate from `struct dimension` (pure geometry). Pass axes via
the multiscale or HCS config.

## Output structure

### Single array
```
out.zarr/
  zarr.json           root group
  0/
    zarr.json         array metadata
    c/0/0             shard data
```

### Multiscale
```
out.zarr/
  zarr.json           root group
  multiscale/
    zarr.json         NGFF group (multiscales attribute)
    0/
      zarr.json       L0 array
      c/...           L0 shards
    1/
      zarr.json       L1 array (downsampled)
      c/...           L1 shards
```

### HCS
```
plate.zarr/
  zarr.json           root group
  plate/
    zarr.json         plate group (OME plate attributes)
    A/
      zarr.json       row group
      1/
        zarr.json     well group (OME well attributes)
        0/
          zarr.json   FOV multiscale group
          0/
            zarr.json L0 array
            c/...     L0 shards
```

## Headers

| Header | Purpose |
|---|---|
| `zarr/store.h` | Abstract store interface |
| `zarr/store_fs.h` | Filesystem store |
| `zarr/store_s3.h` | S3 store |
| `zarr/shard_pool.h` | Abstract shard pool interface |
| `zarr/zarr_array.h` | Zarr v3 array format layer |
| `zarr/zarr_group.h` | Group node write utility |
| `ngff/ngff_axis.h` | NGFF axis metadata types |
| `ngff/ngff_multiscale.h` | NGFF multiscale format layer |
| `hcs/hcs.h` | HCS plate/well/FOV hierarchy |
| `hcs/hcs_metadata.h` | HCS metadata JSON generation |
