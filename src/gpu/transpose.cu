#include "defs.limits.h"
#include "gpu/transpose.h"
#include "util/prelude.h"
#include <stdint.h>
#include <type_traits>

// Tiling the source per block measures faster than one element per thread.
template<typename T>
constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / (int)sizeof(T);

// Riding along in the launch parameters puts the shape and strides in the
// constant bank, so the kernel no longer loads one of each from global memory
// per dimension per element.
template<typename Index>
struct scatter_layout
{
  typedef typename std::make_signed<Index>::type stride_type;

  Index shape[MAX_RANK];
  stride_type strides[MAX_RANK];
  int rank;
  int first_dim; // dimensions before this one never move the destination
};

// Transpose data kernel - v0
template<typename T, typename Index>
__global__ void __launch_bounds__(256, 4)
  transpose_v0_k(T* d_dst,
                 const T* d_src,
                 Index src_size,
                 Index i_offset,
                 Index region_stride,
                 scatter_layout<Index> layout)
{
  const Index block_offset = (Index)blockIdx.x * ELEMENTS_PER_BLOCK<T>;
  const Index left = src_size - block_offset;
  const int elements =
    left < (Index)ELEMENTS_PER_BLOCK<T> ? (int)left : ELEMENTS_PER_BLOCK<T>;

  for (int i = threadIdx.x; i < elements; i += blockDim.x) {
    Index rest = i_offset + block_offset + (Index)i;
    Index out_offset = 0;

    for (int d = layout.rank - 1; d >= layout.first_dim; --d) {
      const Index coord = rest % layout.shape[d];
      rest /= layout.shape[d];
      out_offset += coord * (Index)layout.strides[d];
    }

    // What the decomposition left over counts whole epochs, and each epoch
    // lands in the next region of the destination.
    d_dst[out_offset + rest * region_stride] = d_src[block_offset + i];
  }
}

struct transpose_args
{
  CUdeviceptr d_dst_beg;
  CUdeviceptr d_src_beg;
  uint64_t src_bytes;
  uint64_t i_offset;
  uint64_t epoch_elements;
  uint64_t region_bytes;
  uint8_t rank;
  const uint64_t* shape;
  const int64_t* strides;
  CUstream stream;
};

// The trailing dimensions multiply out to one epoch, so dividing the element
// index by them leaves the epoch number — no separate division by
// epoch_elements, and no iterations for the dimensions in front. Returns the
// dimension the decomposition can stop at, or -1 when the shape does not
// factor that way or a skipped dimension would have moved the destination.
static int
first_decomposed_dim(const struct transpose_args& a)
{
  uint64_t product = 1;
  int first = a.rank;

  while (product != a.epoch_elements) {
    if (first == 0)
      return -1;
    const uint64_t n = a.shape[first - 1];
    if (n == 0 || n > a.epoch_elements / product)
      return -1;
    product *= n;
    --first;
  }

  for (int d = 0; d < first; ++d) {
    if (a.strides[d] != 0 && a.shape[d] != 1)
      return -1;
  }
  return first;
}

static int
add_within_32_bits(uint64_t* total, uint64_t count, uint64_t step)
{
  if (count != 0 && step > (UINT32_MAX - *total) / count)
    return 0;
  *total += count * step;
  return 1;
}

// 32-bit indexing takes the per-element divisions off the 64-bit division
// subroutine, so use it whenever every index the kernel forms fits.
static int
indices_fit_32_bits(const struct transpose_args& a,
                    uint64_t src_size,
                    int first_dim,
                    uint64_t region_stride)
{
  const uint64_t last = a.i_offset + src_size - 1;
  if (last < a.i_offset || last > UINT32_MAX)
    return 0;

  uint64_t max_offset = 0;
  if (!add_within_32_bits(&max_offset, last / a.epoch_elements, region_stride))
    return 0;

  for (int d = first_dim; d < a.rank; ++d) {
    if (a.shape[d] > UINT32_MAX || a.strides[d] < 0)
      return 0;
    if (!add_within_32_bits(
          &max_offset, a.shape[d] - 1, (uint64_t)a.strides[d]))
      return 0;
  }
  return 1;
}

template<typename T, typename Index>
static void
launch(const struct transpose_args& a,
       uint64_t src_size,
       uint64_t region_stride,
       int first_dim)
{
  scatter_layout<Index> layout = {};
  layout.rank = a.rank;
  layout.first_dim = first_dim;
  for (int d = first_dim; d < a.rank; ++d) {
    layout.shape[d] = (Index)a.shape[d];
    layout.strides[d] =
      (typename scatter_layout<Index>::stride_type)a.strides[d];
  }

  const int block_size = 256;
  const unsigned grid_size =
    (unsigned)((src_size + ELEMENTS_PER_BLOCK<T> - 1) / ELEMENTS_PER_BLOCK<T>);

  transpose_v0_k<T, Index>
    <<<grid_size, block_size, 0, (cudaStream_t)a.stream>>>(
      (T*)a.d_dst_beg,
      (const T*)a.d_src_beg,
      (Index)src_size,
      (Index)a.i_offset,
      (Index)region_stride,
      layout);
}

template<typename T>
static int
transpose_launch(const struct transpose_args& a)
{
  const uint64_t src_size = a.src_bytes / sizeof(T);
  if (src_size == 0)
    return 0;

  // Every declaration comes before the checks: a goto may not jump past one.
  const int first_dim = first_decomposed_dim(a);
  const uint64_t region_stride = a.region_bytes / sizeof(T);

  CHECK(Error, a.d_src_beg % sizeof(T) == 0);
  CHECK(Error, a.region_bytes % sizeof(T) == 0);
  CHECK(Error, a.epoch_elements > 0);
  CHECK(Error, a.rank <= MAX_RANK);
  CHECK(Error, first_dim >= 0);

  if (indices_fit_32_bits(a, src_size, first_dim, region_stride))
    launch<T, uint32_t>(a, src_size, region_stride, first_dim);
  else
    launch<T, uint64_t>(a, src_size, region_stride, first_dim);
  return 0;

Error:
  return 1;
}

extern "C" int
transpose(CUdeviceptr d_dst_beg,
          CUdeviceptr d_src_beg,
          uint64_t src_bytes,
          uint8_t bpe,
          uint64_t i_offset,
          uint64_t epoch_elements,
          uint64_t region_bytes,
          uint8_t rank,
          const uint64_t* shape,
          const int64_t* strides,
          CUstream stream)
{
  const struct transpose_args a = { d_dst_beg, d_src_beg,      src_bytes,
                                    i_offset,  epoch_elements, region_bytes,
                                    rank,      shape,          strides,
                                    stream };
  switch (bpe) {
    case 1:
      return transpose_launch<uint8_t>(a);
    case 2:
      return transpose_launch<uint16_t>(a);
    case 4:
      return transpose_launch<uint32_t>(a);
    case 8:
      return transpose_launch<uint64_t>(a);
    default:
      log_error("transpose: unsupported bpe %u", (unsigned)bpe);
      return 1;
  }
}
