# Formats guide

Chucky writes zarr v3 sharded arrays with optional OME-NGFF and HCS metadata.
The format layers compose — each builds on the one below it.

## Architecture

```
hcs_plate       (hcs.h)    HCS plate/well/FOV hierarchy
  |
ngff_multiscale (ngff.h)   OME-NGFF v0.5 multiscale group
  |
zarr_array      (zarr.h)   Zarr v3 sharded array
  |
store           (store.h)  Storage backend (FS or S3)
```

Each layer uses the one below. `hcs_plate` creates `ngff_multiscale`
instances, which create `zarr_array` instances. Pool and shard geometry
are managed internally.

## Quick start

### Single array on filesystem

```c
#include "store.h"
#include "zarr.h"

struct store* store = store_fs_create("/data/out.zarr", 0);

struct zarr_array_config cfg = {
    .data_type = dtype_u16,
    .rank = 2,
    .dimensions = dims,
    .codec = codec,
};

zarr_write_group(store, "zarr.json", NULL);  // root group
struct zarr_array* a = zarr_array_create(store, "0", &cfg);

// Use the shard_sink interface with the stream
struct shard_sink* sink = zarr_array_as_shard_sink(a);

// Cleanup
zarr_array_destroy(a);
store_destroy(store);
```

### OME-NGFF multiscale

```c
#include "ngff.h"
#include "store.h"
#include "zarr.h"

struct store* store = store_fs_create("/data/out.zarr", 0);

zarr_write_group(store, "zarr.json", NULL);  // root group

struct ngff_multiscale_config cfg = {
    .data_type = dtype_u16,
    .rank = 3,
    .dimensions = dims,     // L0 dimensions
    .axes = axes,           // NGFF axis metadata (unit, scale, type)
};

struct ngff_multiscale* ms =
    ngff_multiscale_create(store, "multiscale", &cfg);
struct shard_sink* sink = ngff_multiscale_as_shard_sink(ms);

// Cleanup
ngff_multiscale_destroy(ms);
store_destroy(store);
```

### HCS plate

```c
#include "hcs.h"
#include "store.h"

struct store* store = store_fs_create("/data/plate.zarr", 0);

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

struct hcs_plate* plate = hcs_plate_create(store, &cfg);

// Get shard_sink for well A/1, field 0
struct shard_sink* sink = hcs_plate_fov_sink(plate, 0, 0, 0);

// Cleanup
hcs_plate_destroy(plate);
store_destroy(store);
```

### S3 backend

Replace `store_fs_create` with `store_s3_create` — everything else is
identical:

```c
#include "store.h"

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

`struct store` (`store.h`) is an opaque storage backend. Users create one
via `store_fs_create` or `store_s3_create` and pass it to the format layers.
Destroy with `store_destroy`.

### Geometry

Shard geometry (`shard_counts`, `chunks_per_shard`, `shard_inner_count`)
is computed internally by `zarr_array_create` from the dimension array.
Callers only specify `chunk_size` and `chunks_per_shard` on the
`struct dimension` — the rest is derived.

`ngff_multiscale` computes per-level geometry internally via `lod_plan`.
`hcs_plate` delegates to `ngff_multiscale`.

### NGFF axis metadata

`struct ngff_axis` (`ngff.h`) describes per-dimension metadata:
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

## Public headers

| Header | Purpose |
|---|---|
| `store.h` | Store creation (FS, S3) and destruction |
| `zarr.h` | Zarr v3 array + group write |
| `ngff.h` | NGFF multiscale + axis types |
| `hcs.h` | HCS plate/well/FOV hierarchy + metadata |
