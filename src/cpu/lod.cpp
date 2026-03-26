extern "C"
{
#include "cpu/lod.h"

#include "util/index.ops.h"
#include "util/prelude.h"
}

#include <stdlib.h>
#include <string.h>
#include <type_traits>

// ---- Accumulator type traits ----

template<typename T>
struct reduce_acc
{
  using type = T;
};
template<>
struct reduce_acc<uint8_t>
{
  using type = uint32_t;
};
template<>
struct reduce_acc<uint16_t>
{
  using type = uint32_t;
};
template<>
struct reduce_acc<uint32_t>
{
  using type = uint64_t;
};
template<>
struct reduce_acc<int8_t>
{
  using type = int32_t;
};
template<>
struct reduce_acc<int16_t>
{
  using type = int32_t;
};
template<>
struct reduce_acc<int32_t>
{
  using type = int64_t;
};

// ---- Reduce window ----

template<typename T>
static T
reduce_window(const T* src,
              uint64_t start,
              uint64_t end,
              lod_reduce_method method)
{
  using Acc = typename reduce_acc<T>::type;
  uint64_t len = end - start;

  switch (method) {
    case lod_reduce_mean: {
      Acc sum = 0;
      for (uint64_t j = start; j < end; ++j)
        sum += (Acc)src[j];
      return (T)(sum / (Acc)len);
    }
    case lod_reduce_min: {
      T best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] < best)
          best = src[j];
      return best;
    }
    case lod_reduce_max: {
      T best = src[start];
      for (uint64_t j = start + 1; j < end; ++j)
        if (src[j] > best)
          best = src[j];
      return best;
    }
    case lod_reduce_median: {
      T buf[16];
      uint64_t n = (len <= 16) ? len : 16;
      for (uint64_t j = 0; j < n; ++j)
        buf[j] = src[start + j];
      for (uint64_t i = 1; i < n; ++i) {
        T key = buf[i];
        uint64_t k = i;
        while (k > 0 && buf[k - 1] > key) {
          buf[k] = buf[k - 1];
          --k;
        }
        buf[k] = key;
      }
      return buf[n / 2];
    }
    case lod_reduce_max_suppressed: {
      T t1 = src[start], t2 = src[start];
      for (uint64_t j = start + 1; j < end; ++j) {
        T v = src[j];
        if (v >= t1) {
          t2 = t1;
          t1 = v;
        } else if (v > t2)
          t2 = v;
      }
      return t2;
    }
    case lod_reduce_min_suppressed: {
      T b1 = src[start], b2 = src[start];
      for (uint64_t j = start + 1; j < end; ++j) {
        T v = src[j];
        if (v <= b1) {
          b2 = b1;
          b1 = v;
        } else if (v < b2)
          b2 = v;
      }
      return b2;
    }
  }
  return T{};
}

// ---- Scatter helpers ----

static uint64_t
plan_batch_index(const lod_plan* p, const uint64_t* full_coords)
{
  uint64_t idx = 0, stride = 1;
  for (int k = p->batch_ndim - 1; k >= 0; --k) {
    idx += full_coords[p->batch_map[k]] * stride;
    stride *= p->batch_shape[k];
  }
  return idx;
}

static void
plan_extract_lod(const lod_plan* p,
                 const uint64_t* full_coords,
                 uint64_t* lod_coords)
{
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_coords[k] = full_coords[p->lod_map[k]];
}

// ---- Typed operations ----

template<typename T>
static void
scatter_typed(const lod_plan* p, const T* src, T* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = lod_span_len(lod_spans_at(&p->levels, 0));

#pragma omp parallel for schedule(static) if(n > 1024)
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t full_coords[LOD_MAX_NDIM];
    uint64_t lod_coords[LOD_MAX_NDIM];
    uint64_t rest = i;
    for (int d = p->ndim - 1; d >= 0; --d) {
      full_coords[d] = rest % full_shape[d];
      rest /= full_shape[d];
    }
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_lod(p, full_coords, lod_coords);
    uint64_t pos = morton_rank(p->lod_ndim, p->lod_shapes[0], lod_coords, 0);
    dst[b * p->lod_nelem[0] + pos] = src[i];
  }
}

// ---- Scatter LUT + Gather ----

static void
build_scatter_lut(const lod_plan* p, uint32_t* lut)
{
  const int ndim = p->ndim;
  const int lod_ndim = p->lod_ndim;
  const uint64_t* lod_shape = p->lod_shapes[0];
  const uint64_t lod_count = p->lod_nelem[0];

  // Compute full-shape row-major strides.
  uint64_t full_strides[LOD_MAX_NDIM];
  full_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d)
    full_strides[d] = full_strides[d + 1] * p->shapes[0][d + 1];

  // Precompute LOD strides in the source linear layout.
  uint64_t lod_src_strides[LOD_MAX_NDIM];
  for (int k = 0; k < lod_ndim; ++k)
    lod_src_strides[k] = full_strides[p->lod_map[k]];

#pragma omp parallel for schedule(static) if(lod_count > 1024)
  for (uint64_t gid = 0; gid < lod_count; ++gid) {
    uint64_t coords[LOD_MAX_NDIM];
    uint64_t rest = gid;
    uint64_t src_offset = 0;
    for (int k = lod_ndim - 1; k >= 0; --k) {
      coords[k] = rest % lod_shape[k];
      rest /= lod_shape[k];
      src_offset += coords[k] * lod_src_strides[k];
    }
    uint64_t morton_pos = morton_rank(lod_ndim, lod_shape, coords, 0);
    lut[morton_pos] = (uint32_t)src_offset;
  }
}

static void
build_scatter_batch_offsets(const lod_plan* p, uint64_t* offsets)
{
  const int ndim = p->ndim;

  uint64_t full_strides[LOD_MAX_NDIM];
  full_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d)
    full_strides[d] = full_strides[d + 1] * p->shapes[0][d + 1];

  for (uint64_t b = 0; b < p->batch_count; ++b) {
    uint64_t rest = b;
    uint64_t offset = 0;
    for (int k = p->batch_ndim - 1; k >= 0; --k) {
      uint64_t coord = rest % p->batch_shape[k];
      rest /= p->batch_shape[k];
      offset += coord * full_strides[p->batch_map[k]];
    }
    offsets[b] = offset;
  }
}

template<typename T>
static void
gather_typed(const lod_plan* p,
             const T* src,
             T* dst,
             const uint32_t* scatter_lut,
             const uint64_t* batch_offsets)
{
  const uint64_t lod_count = p->lod_nelem[0];
  const uint64_t batch_count = p->batch_count;

  // Batch-outer: sequential writes per batch, random reads via LUT.
  // The nowait allows threads to start the next batch immediately.
#pragma omp parallel if(lod_count > 1024)
  {
    for (uint64_t b = 0; b < batch_count; ++b) {
      const T* batch_src = src + batch_offsets[b];
      T* batch_dst = dst + b * lod_count;

#pragma omp for schedule(static) nowait
      for (uint64_t m = 0; m < lod_count; ++m)
        batch_dst[m] = batch_src[scatter_lut[m]];
    }
  }
}

template<typename T>
static void
reduce_typed(const lod_plan* p, T* values, lod_reduce_method method)
{
  for (int l = 0; l < p->nlod - 1; ++l) {
    lod_span seg = lod_segment(p, l);
    uint64_t src_ds = p->lod_nelem[l];
    uint64_t dst_ds = p->lod_nelem[l + 1];
    lod_span src_lv = lod_spans_at(&p->levels, l);
    lod_span dst_lv = lod_spans_at(&p->levels, l + 1);

    // Batches and destination elements within a level are independent.
    uint64_t total_work = p->batch_count * dst_ds;
#pragma omp parallel for schedule(static) if(total_work > 1024)
    for (uint64_t wi = 0; wi < total_work; ++wi) {
      uint64_t b = wi / dst_ds;
      uint64_t i = wi % dst_ds;
      uint64_t src_base = src_lv.beg + b * src_ds;
      uint64_t dst_base = dst_lv.beg + b * dst_ds;
      uint64_t start = (i > 0) ? p->ends[seg.beg + i - 1] : 0;
      uint64_t end = p->ends[seg.beg + i];
      values[dst_base + i] =
        reduce_window(values + src_base, start, end, method);
    }
  }
}

template<typename T>
static void
morton_to_chunks_typed(const T* values,
                       T* chunks,
                       const uint32_t* chunk_lut,
                       const uint64_t* batch_offsets,
                       uint64_t lod_count,
                       uint64_t batch_count)
{
#pragma omp parallel if(lod_count > 1024)
  {
    for (uint64_t b = 0; b < batch_count; ++b) {
      const T* batch_values = values + b * lod_count;
      uint64_t batch_base = batch_offsets[b];

#pragma omp for schedule(static) nowait
      for (uint64_t i = 0; i < lod_count; ++i) {
        chunks[batch_base + chunk_lut[i]] = batch_values[i];
      }
    }
  }
}

// ---- Morton-to-chunks LUT ----

static void
build_chunk_lut(const lod_plan* p,
                int lv,
                const tile_stream_layout* layout,
                uint32_t* chunk_lut)
{
  const uint64_t* lod_shape = p->lod_shapes[lv];
  const uint64_t lod_count = p->lod_nelem[lv];
  const int lod_ndim = p->lod_ndim;

#pragma omp parallel for schedule(static) if(lod_count > 1024)
  for (uint64_t gid = 0; gid < lod_count; ++gid) {
    uint64_t coords[LOD_MAX_NDIM];
    int64_t offset = 0;
    uint64_t rest = gid;
    for (int d = lod_ndim - 1; d >= 0; --d) {
      uint64_t coord = rest % lod_shape[d];
      rest /= lod_shape[d];
      coords[d] = coord;

      // LOD dim d maps to full dim p->lod_map[d] in the layout.
      int ld = p->lod_map[d];
      uint64_t chunk_size_d = layout->lifted_shape[2 * ld + 1];
      uint64_t chunk_idx = coord / chunk_size_d;
      uint64_t within = coord % chunk_size_d;
      offset += (int64_t)chunk_idx * layout->lifted_strides[2 * ld];
      offset += (int64_t)within * layout->lifted_strides[2 * ld + 1];
    }

    uint64_t morton_pos = morton_rank(lod_ndim, lod_shape, coords, 0);
    chunk_lut[morton_pos] = (uint32_t)offset;
  }
}

// ---- Dim0 fold/emit ----

// Overflow-safe (a + b) >> s for integer types.
// Float/64-bit: just add (emit handles final division).
template<typename T>
static T
overflow_safe_add_shift(T a, T b, int s)
{
  if constexpr (std::is_floating_point<T>::value || sizeof(T) >= 8)
    return a + b;
  else {
    T mask = (T)((1u << s) - 1);
    return (T)((a >> s) + (b >> s) + (((a & mask) + (b & mask)) >> s));
  }
}

template<typename T>
static void
dim0_fold_typed(T* accum,
                const T* new_data,
                uint64_t n,
                uint32_t count,
                int level,
                lod_reduce_method method)
{
  if (count == 0) {
#pragma omp parallel for schedule(static) if(n > 1024)
    for (uint64_t i = 0; i < n; ++i)
      accum[i] = new_data[i];
    return;
  }
  switch (method) {
    case lod_reduce_mean:
#pragma omp parallel for schedule(static) if(n > 1024)
      for (uint64_t i = 0; i < n; ++i)
        accum[i] = overflow_safe_add_shift(accum[i], new_data[i], level);
      break;
    case lod_reduce_min:
#pragma omp parallel for schedule(static) if(n > 1024)
      for (uint64_t i = 0; i < n; ++i)
        if (new_data[i] < accum[i])
          accum[i] = new_data[i];
      break;
    case lod_reduce_max:
#pragma omp parallel for schedule(static) if(n > 1024)
      for (uint64_t i = 0; i < n; ++i)
        if (new_data[i] > accum[i])
          accum[i] = new_data[i];
      break;
    default:
      break;
  }
}

template<typename T>
static void
dim0_emit_typed(T* dst,
                const T* accum,
                uint64_t n,
                uint32_t count,
                lod_reduce_method method)
{
  if constexpr (std::is_floating_point<T>::value) {
    if (method == lod_reduce_mean) {
      T divisor = (T)count;
#pragma omp parallel for schedule(static) if(n > 1024)
      for (uint64_t i = 0; i < n; ++i)
        dst[i] = accum[i] / divisor;
      return;
    }
  }
  // int mean (pre-divided), min, max: just copy
#pragma omp parallel for schedule(static) if(n > 1024)
  for (uint64_t i = 0; i < n; ++i)
    dst[i] = accum[i];
}

// ---- Dispatch macro (only used in the public API) ----

#define DISPATCH(dtype, call)                                                  \
  switch (dtype) {                                                             \
    case dtype_u8:                                                         \
      call(uint8_t);                                                           \
      break;                                                                   \
    case dtype_u16:                                                        \
      call(uint16_t);                                                          \
      break;                                                                   \
    case dtype_u32:                                                        \
      call(uint32_t);                                                          \
      break;                                                                   \
    case dtype_u64:                                                        \
      call(uint64_t);                                                          \
      break;                                                                   \
    case dtype_i8:                                                         \
      call(int8_t);                                                            \
      break;                                                                   \
    case dtype_i16:                                                        \
      call(int16_t);                                                           \
      break;                                                                   \
    case dtype_i32:                                                        \
      call(int32_t);                                                           \
      break;                                                                   \
    case dtype_i64:                                                        \
      call(int64_t);                                                           \
      break;                                                                   \
    case dtype_f32:                                                        \
      call(float);                                                             \
      break;                                                                   \
    case dtype_f64:                                                        \
      call(double);                                                            \
      break;                                                                   \
    default:                                                                   \
      return 1;                                                                \
  }

// ---- Public API (extern "C") ----

extern "C" int
lod_cpu_reduce(const lod_plan* p,
               void* values,
               enum dtype dtype,
               lod_reduce_method method)
{
#define DO(T) reduce_typed(p, (T*)values, method)
  DISPATCH(dtype, DO);
#undef DO
  return 0;
}

extern "C" void
lod_cpu_build_chunk_lut(const lod_plan* p,
                        int lv,
                        const tile_stream_layout* layout,
                        uint32_t* chunk_lut)
{
  build_chunk_lut(p, lv, layout, chunk_lut);
}

extern "C" int
lod_cpu_morton_to_chunks(const lod_plan* p,
                         const void* values,
                         void* chunk_pool,
                         int lv,
                         const tile_stream_layout* layout,
                         const uint32_t* chunk_lut_in,
                         const uint64_t* batch_chunk_offsets,
                         enum dtype dtype)
{
  const uint64_t lod_count = p->lod_nelem[lv];
  const lod_span lv_span = lod_spans_at(&p->levels, (uint64_t)lv);
  const size_t bytes_per_element = dtype_bpe(dtype);
  const char* lv_values = (const char*)values + lv_span.beg * bytes_per_element;

  // Use provided LUT or build one (legacy/standalone path).
  uint32_t* chunk_lut_alloc = NULL;
  const uint32_t* chunk_lut = chunk_lut_in;
  if (!chunk_lut) {
    chunk_lut_alloc = (uint32_t*)malloc(lod_count * sizeof(uint32_t));
    if (!chunk_lut_alloc)
      return 1;
    build_chunk_lut(p, lv, layout, chunk_lut_alloc);
    chunk_lut = chunk_lut_alloc;
  }

#define DO(T)                                                                  \
  morton_to_chunks_typed((const T*)lv_values, (T*)chunk_pool, chunk_lut,       \
                         batch_chunk_offsets, lod_count, p->batch_count)
  DISPATCH(dtype, DO);
#undef DO

  free(chunk_lut_alloc);
  return 0;
}

extern "C" int
lod_cpu_append_fold(const lod_plan* p,
                  const void* morton_values,
                  void* accum,
                  const uint32_t* counts,
                  enum dtype dtype,
                  lod_reduce_method method)
{
  const size_t bytes_per_element = dtype_bpe(dtype);
  uint64_t accum_offset = 0;

  for (int lv = 1; lv < p->nlod; ++lv) {
    lod_span lev = lod_spans_at(&p->levels, (uint64_t)lv);
    uint64_t n = p->batch_count * p->lod_nelem[lv];
    const char* src = (const char*)morton_values + lev.beg * bytes_per_element;
    char* dst = (char*)accum + accum_offset * bytes_per_element;

#define DO(T)                                                                  \
  dim0_fold_typed((T*)dst, (const T*)src, n, counts[lv], lv, method)
    DISPATCH(dtype, DO);
#undef DO

    accum_offset += n;
  }

  return 0;
}

extern "C" int
lod_cpu_append_emit(const lod_plan* p,
                  void* morton_values,
                  const void* accum,
                  int lv,
                  uint32_t count,
                  enum dtype dtype,
                  lod_reduce_method method)
{
  const size_t bytes_per_element = dtype_bpe(dtype);
  lod_span lev = lod_spans_at(&p->levels, (uint64_t)lv);
  uint64_t n = p->batch_count * p->lod_nelem[lv];

  // Compute accum offset for this level
  uint64_t accum_offset = 0;
  for (int k = 1; k < lv; ++k)
    accum_offset += p->batch_count * p->lod_nelem[k];

  char* dst = (char*)morton_values + lev.beg * bytes_per_element;
  const char* src = (const char*)accum + accum_offset * bytes_per_element;

#define DO(T) dim0_emit_typed((T*)dst, (const T*)src, n, count, method)
  DISPATCH(dtype, DO);
#undef DO

  return 0;
}

extern "C" void
lod_cpu_build_scatter_lut(const lod_plan* p, uint32_t* lut)
{
  build_scatter_lut(p, lut);
}

extern "C" void
lod_cpu_build_scatter_batch_offsets(const lod_plan* p, uint64_t* offsets)
{
  build_scatter_batch_offsets(p, offsets);
}

extern "C" int
lod_cpu_gather(const lod_plan* p,
               const void* src,
               void* dst,
               const uint32_t* scatter_lut,
               const uint64_t* batch_offsets,
               enum dtype dtype)
{
#define DO(T)                                                                  \
  gather_typed(p, (const T*)src, (T*)dst, scatter_lut, batch_offsets)
  DISPATCH(dtype, DO);
#undef DO
  return 0;
}
