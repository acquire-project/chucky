#include "defs.limits.h"
#include "gpu/prelude.cuda.h"
#include "gpu/transpose.h"
#include "util/prelude.h"
#include <stdint.h>

// Tiling the source per block measures faster than one element per thread.
template<typename T>
constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / (int)sizeof(T);

// Travels in the launch parameters, so the kernel reads it from the constant
// bank instead of paying a load per dimension per element.
template<typename Index>
struct scatter_layout
{
  Index shape[MAX_RANK];
  Index strides[MAX_RANK];
  int ndims;
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

    for (int d = layout.ndims - 1; d >= 0; --d) {
      const Index coord = rest % layout.shape[d];
      rest /= layout.shape[d];
      out_offset += coord * layout.strides[d];
    }

    // The visited extents span one epoch, so what is left counts epochs.
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

template<typename T>
static int
args_valid(const struct transpose_args& a)
{
  CHECK(Invalid, a.rank <= MAX_RANK);
  CHECK(Invalid, a.d_src_beg % sizeof(T) == 0);
  CHECK(Invalid, a.region_bytes % sizeof(T) == 0);
  CHECK(Invalid, a.epoch_elements > 0);
  return 1;

Invalid:
  return 0;
}

// Finds where to start decomposing an element index so that what the
// decomposition leaves behind is the epoch number. Returns -1 when no run of
// trailing extents makes an epoch, or when a dimension it would skip can still
// reach the destination.
static int
first_decomposed_dim(uint64_t epoch_elements,
                     uint8_t rank,
                     const uint64_t* shape,
                     const int64_t* strides)
{
  uint64_t product = 1;
  int first = rank;

  while (product != epoch_elements) {
    if (first == 0)
      goto NoEpoch;
    const uint64_t n = shape[first - 1];
    if (n == 0 || n > epoch_elements / product)
      goto NoEpoch;
    product *= n;
    --first;
  }

  for (int d = 0; d < first; ++d) {
    if (strides[d] != 0 && shape[d] != 1) {
      log_error("transpose: lifted dimension %d lies outside an epoch, but its "
                "extent %llu and stride %lld still reach the destination",
                d,
                (unsigned long long)shape[d],
                (long long)strides[d]);
      return -1;
    }
  }
  return first;

NoEpoch:
  log_error("transpose: no run of trailing extents multiplies out to an epoch "
            "of %llu elements",
            (unsigned long long)epoch_elements);
  return -1;
}

extern "C" int
transpose_check_layout(uint64_t epoch_elements,
                       uint8_t rank,
                       const uint64_t* shape,
                       const int64_t* strides)
{
  CHECK(Invalid, rank <= MAX_RANK);
  CHECK_SILENT(Invalid,
               first_decomposed_dim(epoch_elements, rank, shape, strides) >= 0);
  return 0;

Invalid:
  return 1;
}

static int
add_within_32_bits(uint64_t* total, uint64_t count, uint64_t step)
{
  if (count != 0 && step > (UINT32_MAX - *total) / count)
    return 0;
  *total += count * step;
  return 1;
}

// 32-bit division inlines where the 64-bit routine does not, so narrow indexing
// is worth taking wherever every index the kernel forms fits.
static int
indices_fit_32_bits(const struct transpose_args& a,
                    uint64_t src_size,
                    int first_dim,
                    uint64_t region_stride)
{
  const uint64_t last = a.i_offset + src_size - 1;
  if (src_size > UINT32_MAX || last < a.i_offset || last > UINT32_MAX)
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
static int
launch(const struct transpose_args& a, int first_dim)
{
  const uint64_t src_size = a.src_bytes / sizeof(T);
  const uint64_t region_stride = a.region_bytes / sizeof(T);

  scatter_layout<Index> layout = {};
  layout.ndims = a.rank - first_dim;
  for (int d = 0; d < layout.ndims; ++d) {
    layout.shape[d] = (Index)a.shape[first_dim + d];
    layout.strides[d] = (Index)a.strides[first_dim + d];
  }

  const int block_size = 256;
  const unsigned grid_size =
    (unsigned)ceildiv(src_size, (uint64_t)ELEMENTS_PER_BLOCK<T>);

  return CUDA_LAUNCH(transpose_v0_k<T, Index>
                     <<<grid_size, block_size, 0, (cudaStream_t)a.stream>>>(
                       (T*)a.d_dst_beg,
                       (const T*)a.d_src_beg,
                       (Index)src_size,
                       (Index)a.i_offset,
                       (Index)region_stride,
                       layout));
}

template<typename T>
static int
transpose_launch(const struct transpose_args& a)
{
  const uint64_t src_size = a.src_bytes / sizeof(T);
  if (src_size == 0)
    return 0;
  if (!args_valid<T>(a))
    return 1;

  const int first_dim =
    first_decomposed_dim(a.epoch_elements, a.rank, a.shape, a.strides);
  if (first_dim < 0)
    return 1;

  if (indices_fit_32_bits(a, src_size, first_dim, a.region_bytes / sizeof(T)))
    return launch<T, uint32_t>(a, first_dim);
  return launch<T, uint64_t>(a, first_dim);
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
