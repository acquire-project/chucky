#pragma once

#include <stddef.h>
#include <stdint.h>

// CPU scatter transpose using the vadd() algorithm.
// Scatters src_bytes/bpe elements from src into dst using lifted shape/strides.
// i_offset is the global flat input offset (for multi-call accumulation).
// Returns 0 on success, non-zero on allocation failure.
int transpose_cpu(void* dst,
                  const void* src,
                  uint64_t src_bytes,
                  uint8_t bpe,
                  uint64_t i_offset,
                  uint8_t lifted_rank,
                  const uint64_t* lifted_shape,
                  const int64_t* lifted_strides);
