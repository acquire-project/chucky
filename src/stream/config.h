#pragma once

#include "stream/layouts.h"
#include "types.stream.h"

// Validate config and compute all CPU-only layout math.
// codec_alignment: input chunk alignment required by the compression backend
//   (nvcomp needs 512+, CPU libs need 1).
// max_output_size_fn: returns max compressed bytes for a given
// codec+chunk_bytes
//   (backend-specific — nvcomp vs CPU lz4/zstd bounds may differ).
// shard_alignment: required write alignment for the I/O backend (e.g. page
//   size for O_DIRECT). 0 = no alignment constraint.
// Returns 0 on success.
int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       size_t codec_alignment,
                       size_t (*max_output_size_fn)(enum compression_codec,
                                                    size_t chunk_bytes),
                       size_t shard_alignment,
                       struct computed_stream_layouts* out);

// One level's layout, host-side. level_shape is that level's shape in
// elements, alignment is how far each chunk is padded, and storage_order is
// the forward permutation: entry j is the dimension stored at position j.
// Returns 0 on success, 1 on overflow.
int
compute_level_layout(struct tile_stream_layout* layout,
                     uint8_t rank,
                     uint8_t n_append,
                     size_t bytes_per_element,
                     const struct dimension* dims,
                     const uint64_t* level_shape,
                     size_t alignment,
                     const uint8_t* storage_order);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);

// Create a named stream_metric with best_ms initialized to a large value.
struct stream_metric
mk_stream_metric(const char* name, enum metric_owner owner);
