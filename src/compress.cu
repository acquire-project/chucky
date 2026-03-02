#include "compress.h"
#include "log/log.h"
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>
#include <string.h>

static const char*
nvcomp_status_name(nvcompStatus_t st)
{
  switch (st) {
    case nvcompSuccess:
      return "nvcompSuccess";
    case nvcompErrorInvalidValue:
      return "nvcompErrorInvalidValue";
    case nvcompErrorNotSupported:
      return "nvcompErrorNotSupported";
    case nvcompErrorCannotDecompress:
      return "nvcompErrorCannotDecompress";
    case nvcompErrorBadChecksum:
      return "nvcompErrorBadChecksum";
    case nvcompErrorCannotVerifyChecksums:
      return "nvcompErrorCannotVerifyChecksums";
    case nvcompErrorCudaError:
      return "nvcompErrorCudaError";
    case nvcompErrorInternal:
      return "nvcompErrorInternal";
    default:
      return "nvcompUnknown";
  }
}

static int
handle_nvcomp(int level,
              nvcompStatus_t st,
              const char* file,
              int line,
              const char* expr)
{
  if (st == nvcompSuccess)
    return 0;
  log_log(level,
          file,
          line,
          "nvcomp error: %s (%d) %s",
          nvcomp_status_name(st),
          (int)st,
          expr);
  return 1;
}

#define NVCOMP(lbl, e)                                                         \
  do {                                                                         \
    nvcompStatus_t st_ = (e);                                                  \
    if (st_ != nvcompSuccess &&                                                \
        handle_nvcomp(LOG_ERROR, st_, __FILE__, __LINE__, #e)) {               \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define CU(lbl, e)                                                             \
  do {                                                                         \
    CUresult r_ = (e);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s_ = NULL;                                                   \
      cuGetErrorString(r_, &s_);                                               \
      log_error("CUDA error: %s at %s:%d", s_ ? s_ : "unknown",               \
                __FILE__, __LINE__);                                           \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

// --- fill_ptrs kernel ---
// Fills d_ptrs[0..batch_size-1] = base + i * stride (uncomp pointers)
// Fills d_ptrs[batch_size..2*batch_size-1] = base + i * stride (comp pointers)

__global__ void
fill_ptrs_kernel(void** d_ptrs,
                 const char* uncomp_base,
                 size_t uncomp_stride,
                 char* comp_base,
                 size_t comp_stride,
                 size_t batch_size)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    d_ptrs[i] = (void*)(uncomp_base + i * uncomp_stride);
    d_ptrs[batch_size + i] = (void*)(comp_base + i * comp_stride);
  }
}

// --- codec_alignment ---

extern "C" size_t
codec_alignment(enum compression_codec type)
{
  switch (type) {
    case CODEC_LZ4:
      return nvcompLZ4RequiredCompressionAlignment;
    case CODEC_ZSTD:
      return nvcompZstdRequiredCompressionAlignment;
    default:
      return 1;
  }
}

// --- codec_max_output_size ---

extern "C" size_t
codec_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  size_t max_comp = 0;
  switch (type) {
    case CODEC_NONE:
      return chunk_bytes;
    case CODEC_LZ4:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedLZ4CompressDefaultOpts, &max_comp));
      return max_comp;
    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedZstdCompressDefaultOpts, &max_comp));
      return max_comp;
    default:
      break;
  }
Fail:
  return 0;
}

// --- codec_temp_bytes ---

extern "C" size_t
codec_temp_bytes(enum compression_codec type,
                 size_t chunk_bytes,
                 size_t batch_size)
{
  size_t temp = 0;
  switch (type) {
    case CODEC_NONE:
      return 0;
    case CODEC_LZ4:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetTempSizeAsync(
               batch_size,
               chunk_bytes,
               nvcompBatchedLZ4CompressDefaultOpts,
               &temp,
               batch_size * chunk_bytes));
      return temp;
    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetTempSizeAsync(
               batch_size,
               chunk_bytes,
               nvcompBatchedZstdCompressDefaultOpts,
               &temp,
               batch_size * chunk_bytes));
      return temp;
    default:
      break;
  }
Fail:
  return 0;
}

// --- codec_init ---

extern "C" int
codec_init(struct codec* c,
           enum compression_codec type,
           size_t chunk_size,
           size_t batch_size)
{
  memset(c, 0, sizeof(*c));
  c->type = type;
  c->chunk_size = chunk_size;
  c->batch_size = batch_size;
  c->alignment = codec_alignment(type);

  switch (type) {
    case CODEC_NONE:
      c->max_output_size = chunk_size;
      c->temp_bytes = 0;
      break;

    case CODEC_LZ4: {
      size_t max_comp = 0;
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
               chunk_size, nvcompBatchedLZ4CompressDefaultOpts, &max_comp));
      c->max_output_size = max_comp;
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetTempSizeAsync(
               batch_size,
               chunk_size,
               nvcompBatchedLZ4CompressDefaultOpts,
               &c->temp_bytes,
               batch_size * chunk_size));
      break;
    }

    case CODEC_ZSTD: {
      size_t max_comp = 0;
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetMaxOutputChunkSize(
               chunk_size, nvcompBatchedZstdCompressDefaultOpts, &max_comp));
      c->max_output_size = max_comp;
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetTempSizeAsync(
               batch_size,
               chunk_size,
               nvcompBatchedZstdCompressDefaultOpts,
               &c->temp_bytes,
               batch_size * chunk_size));
      break;
    }

    default:
      goto Fail;
  }

  // Allocate device arrays
  CU(Fail,
     cuMemAlloc((CUdeviceptr*)&c->d_comp_sizes, batch_size * sizeof(size_t)));
  CU(Fail,
     cuMemAlloc((CUdeviceptr*)&c->d_uncomp_sizes, batch_size * sizeof(size_t)));

  // Pre-fill d_uncomp_sizes with chunk_size
  {
    size_t* h = (size_t*)malloc(batch_size * sizeof(size_t));
    if (!h)
      goto Fail;
    for (size_t i = 0; i < batch_size; ++i)
      h[i] = chunk_size;
    CUresult rc =
      cuMemcpyHtoD((CUdeviceptr)c->d_uncomp_sizes, h, batch_size * sizeof(size_t));
    free(h);
    CU(Fail, rc);
  }

  // For CODEC_NONE, pre-fill d_comp_sizes with chunk_size
  if (type == CODEC_NONE) {
    size_t* h = (size_t*)malloc(batch_size * sizeof(size_t));
    if (!h)
      goto Fail;
    for (size_t i = 0; i < batch_size; ++i)
      h[i] = chunk_size;
    CUresult rc =
      cuMemcpyHtoD((CUdeviceptr)c->d_comp_sizes, h, batch_size * sizeof(size_t));
    free(h);
    CU(Fail, rc);
  }

  // Pointer arrays for nvcomp (not needed for CODEC_NONE)
  if (type != CODEC_NONE) {
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&c->d_ptrs, 2 * batch_size * sizeof(void*)));
  }

  // Temp workspace
  if (c->temp_bytes > 0) {
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_temp, c->temp_bytes));
  }

  return 1;

Fail:
  codec_free(c);
  return 0;
}

// --- codec_free ---

extern "C" void
codec_free(struct codec* c)
{
  if (c->d_comp_sizes) {
    cuMemFree((CUdeviceptr)c->d_comp_sizes);
    c->d_comp_sizes = NULL;
  }
  if (c->d_uncomp_sizes) {
    cuMemFree((CUdeviceptr)c->d_uncomp_sizes);
    c->d_uncomp_sizes = NULL;
  }
  if (c->d_ptrs) {
    cuMemFree((CUdeviceptr)c->d_ptrs);
    c->d_ptrs = NULL;
  }
  if (c->d_temp) {
    cuMemFree((CUdeviceptr)c->d_temp);
    c->d_temp = NULL;
  }
}

// --- codec_compress ---

extern "C" int
codec_compress(struct codec* c,
               const void* d_input,
               size_t input_stride,
               void* d_output,
               CUstream stream)
{
  const void* const* uncomp_ptrs = (const void* const*)c->d_ptrs;
  void* const* comp_ptrs = (void* const*)(c->d_ptrs + c->batch_size);
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  if (c->type == CODEC_NONE) {
    CU(Fail,
       cuMemcpyDtoDAsync((CUdeviceptr)d_output,
                         (CUdeviceptr)d_input,
                         c->batch_size * c->chunk_size,
                         stream));
    return 1;
  }

  // Fill pointer arrays
  {
    size_t n = c->batch_size;
    unsigned blocks = (unsigned)((n + 255) / 256);
    fill_ptrs_kernel<<<blocks, 256, 0, cuda_stream>>>(
      c->d_ptrs,
      (const char*)d_input,
      input_stride,
      (char*)d_output,
      c->max_output_size,
      n);
  }

  switch (c->type) {
    case CODEC_LZ4:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressAsync(uncomp_ptrs,
                                           c->d_uncomp_sizes,
                                           c->chunk_size,
                                           c->batch_size,
                                           c->d_temp,
                                           c->temp_bytes,
                                           comp_ptrs,
                                           c->d_comp_sizes,
                                           nvcompBatchedLZ4CompressDefaultOpts,
                                           NULL,
                                           cuda_stream));
      break;

    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressAsync(uncomp_ptrs,
                                            c->d_uncomp_sizes,
                                            c->chunk_size,
                                            c->batch_size,
                                            c->d_temp,
                                            c->temp_bytes,
                                            comp_ptrs,
                                            c->d_comp_sizes,
                                            nvcompBatchedZstdCompressDefaultOpts,
                                            NULL,
                                            cuda_stream));
      break;

    default:
      goto Fail;
  }

  return 1;

Fail:
  return 0;
}
