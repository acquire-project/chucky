#pragma once

#include "types.stream.h"

// Validate config and compute all CPU-only layout math.
// codec_alignment: input chunk alignment required by the compression backend
//   (nvcomp needs 512+, CPU libs need 1).
// max_output_size_fn: returns max compressed bytes for a given codec+chunk_bytes
//   (backend-specific — nvcomp vs CPU lz4/zstd bounds may differ).
// Returns 0 on success.
int
compute_stream_layouts(
  const struct tile_stream_configuration* config,
  size_t codec_alignment,
  size_t (*max_output_size_fn)(enum compression_codec, size_t chunk_bytes),
  struct computed_stream_layouts* out);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);

// Create a named stream_metric with best_ms initialized to a large value.
struct stream_metric
mk_stream_metric(const char* name);
