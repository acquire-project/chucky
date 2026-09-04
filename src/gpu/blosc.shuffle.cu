#include "gpu/blosc.shuffle.h"
#include "gpu/prelude.cuda.h"

__global__ static void
shuffle_kernel(const unsigned char* src,
               size_t src_stride,
               unsigned char* dst,
               size_t dst_stride,
               size_t chunk_bytes,
               size_t typesize,
               size_t batch_size)
{
  size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch_size * chunk_bytes;
  if (p >= total)
    return;

  const size_t chunk = p / chunk_bytes;
  const size_t out = p - chunk * chunk_bytes;
  const size_t nelem = chunk_bytes / typesize;
  const size_t complete = nelem * typesize;
  size_t in = out;
  if (out < complete) {
    const size_t byte = out / nelem;
    const size_t elem = out - byte * nelem;
    in = elem * typesize + byte;
  }
  dst[chunk * dst_stride + out] = src[chunk * src_stride + in];
}

extern "C" int
gpu_blosc_shuffle_async(const void* src,
                        size_t src_stride,
                        void* dst,
                        size_t dst_stride,
                        size_t chunk_bytes,
                        size_t typesize,
                        size_t batch_size,
                        CUstream stream)
{
  const size_t total = batch_size * chunk_bytes;
  const unsigned blocks = (unsigned)((total + 255) / 256);
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  return CUDA_LAUNCH(
    shuffle_kernel<<<blocks, 256, 0, cuda_stream>>>((const unsigned char*)src,
                                                    src_stride,
                                                    (unsigned char*)dst,
                                                    dst_stride,
                                                    chunk_bytes,
                                                    typesize,
                                                    batch_size));
}
