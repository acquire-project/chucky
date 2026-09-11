#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct threadpool;
  struct tile_stream_layout;

  // CPU scatter transpose using the vadd() algorithm.
  // Copies directly when the layout proves that the epoch is contiguous;
  // otherwise scatters src_bytes/bpe elements using its lifted shape/strides.
  // i_offset is the global flat input offset (for multi-call accumulation).
  // Returns 0 on success and nonzero for an unsupported element size.
  int transpose_cpu(void* dst,
                    const void* src,
                    uint64_t src_bytes,
                    uint8_t bpe,
                    uint64_t i_offset,
                    const struct tile_stream_layout* layout,
                    struct threadpool* pool);

#ifdef __cplusplus
}
#endif
