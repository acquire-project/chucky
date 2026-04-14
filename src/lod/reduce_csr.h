#pragma once

#include <stdint.h>

struct lod_plan;
struct src_map;

// Host CSR reduce LUT for one level transition (l -> l+1).
// Flattened: batch_count=1, indices contain absolute offsets within the source
// level. Layout matches the scatter kernel's ascending-d fixed-dim enumeration.
//
// Lifecycle:
//   reduce_csr_alloc  — allocates starts, indices, and internal scratch.
//   reduce_csr_build  — fills starts/indices (no allocation).
//   reduce_csr_free   — frees everything, null-safe.
struct reduce_csr
{
  // Outputs.
  uint64_t* starts;  // [dst_segment_size + 1]
  uint64_t* indices; // [src_lod_count]

  // Geometry (set by alloc).
  uint64_t dst_segment_size; // dst fixed_dims_count * dst lod_nelem
  uint64_t src_lod_count;    // src fixed_dims_count * src lod_nelem
  uint64_t batch_count;      // always 1 in current code

  // Scratch (used only during build).
  struct src_map* scratch_map; // [src_lod_count]
  uint64_t* scratch_wpos;      // [dst_segment_size]
};

// Allocate arrays sized by src_total (source-level element count) and
// dst_total (destination-level element count). If either is zero, allocates
// nothing and returns success; build is a no-op in that case.
//
// On failure leaves *csr in a state safe to pass to reduce_csr_free.
// Returns 0 on success, non-zero on failure.
int
reduce_csr_alloc(struct reduce_csr* csr, uint64_t src_total, uint64_t dst_total);

// Compute starts/indices for the transition plan->levels.level[level] ->
// plan->levels.level[level+1]. Assumes alloc has been called with matching
// sizes. No allocation; safe to call repeatedly.
// Returns 0 on success, non-zero on failure.
int
reduce_csr_build(struct reduce_csr* csr,
                 const struct lod_plan* plan,
                 int level);

// Null-safe. Frees all allocations and zeros the struct. Safe to call on a
// zeroed struct, a half-constructed struct (after a failed alloc), or twice.
void
reduce_csr_free(struct reduce_csr* csr);
