#ifndef MORTON_UTIL_H
#define MORTON_UTIL_H

#include "lod.h"
#include "lod_plan.h"

#include <stdint.h>

// CPU reference: scatter src into morton-ordered LOD buffer (f32).
void
lod_scatter_cpu(const struct lod_plan* p, const float* src, float* dst);

// CPU reference: reduce across LOD levels (f32).
void
lod_reduce_cpu(const struct lod_plan* p,
               float* values,
               enum lod_reduce_method method);

// CPU reference: scatter + reduce, allocates output (f32).
// Returns truthy on success (used with CHECK).
int
lod_compute(const struct lod_plan* p,
            const float* src,
            float** out_values,
            enum lod_reduce_method method);

// CPU reference: scatter src into morton-ordered LOD buffer (u16).
void
lod_scatter_cpu_u16(const struct lod_plan* p,
                    const uint16_t* src,
                    uint16_t* dst);

// CPU reference: reduce a single window (u16).
uint16_t
reduce_window_u16(const uint16_t* src,
                  uint64_t start,
                  uint64_t end,
                  enum lod_reduce_method method);

// CPU reference: reduce across LOD levels (u16).
void
lod_reduce_cpu_u16(const struct lod_plan* p,
                   uint16_t* values,
                   enum lod_reduce_method method);

// CPU reference: scatter + reduce, allocates output (u16).
// Returns truthy on success (used with CHECK).
int
lod_compute_u16(const struct lod_plan* p,
                const uint16_t* src,
                uint16_t** out_values,
                enum lod_reduce_method method);

#endif // MORTON_UTIL_H
