#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "log/log.h"
#include <nvcomp/lz4.h>
#include <nvcomp/shared_types.h>
#include <nvcomp/zstd.h>
#include <string.h>

static const char*
nvcomp_status_name(nvcompStatus_t st)
{
#define CASE(e)                                                                \
  case e:                                                                      \
    return #e

  switch (st) {
    CASE(nvcompSuccess);
    CASE(nvcompErrorInvalidValue);
    CASE(nvcompErrorNotSupported);
    CASE(nvcompErrorCannotDecompress);
    CASE(nvcompErrorBadChecksum);
    CASE(nvcompErrorCannotVerifyChecksums);
    CASE(nvcompErrorCudaError);
    CASE(nvcompErrorInternal);
#undef CASE
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
           size_t chunk_bytes,
           size_t batch_size)
{
  memset(c, 0, sizeof(*c));
  c->type = type;
  c->chunk_bytes = chunk_bytes;
  c->batch_size = batch_size;
  c->alignment = codec_alignment(type);

  switch (type) {
    case CODEC_NONE:
      c->max_output_size = chunk_bytes;
      c->temp_bytes = 0;
      break;

    case CODEC_LZ4: {
      size_t max_comp = 0;
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedLZ4CompressDefaultOpts, &max_comp));
      c->max_output_size = max_comp;
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetTempSizeAsync(
               batch_size,
               chunk_bytes,
               nvcompBatchedLZ4CompressDefaultOpts,
               &c->temp_bytes,
               batch_size * chunk_bytes));
      break;
    }

    case CODEC_ZSTD: {
      size_t max_comp = 0;
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedZstdCompressDefaultOpts, &max_comp));
      c->max_output_size = max_comp;
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetTempSizeAsync(
               batch_size,
               chunk_bytes,
               nvcompBatchedZstdCompressDefaultOpts,
               &c->temp_bytes,
               batch_size * chunk_bytes));
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

  // Pre-fill d_uncomp_sizes with chunk_bytes
  {
    size_t* h = (size_t*)malloc(batch_size * sizeof(size_t));
    if (!h)
      goto Fail;
    for (size_t i = 0; i < batch_size; ++i)
      h[i] = chunk_bytes;
    CUresult rc = cuMemcpyHtoD(
      (CUdeviceptr)c->d_uncomp_sizes, h, batch_size * sizeof(size_t));
    free(h);
    CU(Fail, rc);
  }

  // For CODEC_NONE, pre-fill d_comp_sizes with chunk_bytes
  if (type == CODEC_NONE) {
    size_t* h = (size_t*)malloc(batch_size * sizeof(size_t));
    if (!h)
      goto Fail;
    for (size_t i = 0; i < batch_size; ++i)
      h[i] = chunk_bytes;
    CUresult rc = cuMemcpyHtoD(
      (CUdeviceptr)c->d_comp_sizes, h, batch_size * sizeof(size_t));
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

  return 0;

Fail:
  codec_free(c);
  return 1;
}

// --- codec_free ---

extern "C" void
codec_free(struct codec* c)
{
  cu_mem_free((CUdeviceptr)c->d_comp_sizes);
  cu_mem_free((CUdeviceptr)c->d_uncomp_sizes);
  cu_mem_free((CUdeviceptr)c->d_ptrs);
  cu_mem_free((CUdeviceptr)c->d_temp);
  c->d_comp_sizes = NULL;
  c->d_uncomp_sizes = NULL;
  c->d_ptrs = NULL;
  c->d_temp = NULL;
}

// --- codec_compress ---

extern "C" int
codec_compress(struct codec* c,
               const void* d_input,
               size_t input_stride,
               void* d_output,
               size_t actual_batch_size,
               CUstream stream)
{
  size_t n = actual_batch_size ? actual_batch_size : c->batch_size;
  const void* const* uncomp_ptrs = (const void* const*)c->d_ptrs;
  void* const* comp_ptrs = (void* const*)(c->d_ptrs + n);
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  if (c->type == CODEC_NONE) {
    // All callers pass input_stride == chunk_bytes.
    // The strided path is dead code, but kept as a defensive fallback.
    if (input_stride != c->chunk_bytes) {
      for (size_t i = 0; i < n; ++i) {
        CU(Fail,
           cuMemcpyDtoDAsync(
             (CUdeviceptr)((char*)d_output + i * c->chunk_bytes),
             (CUdeviceptr)((const char*)d_input + i * input_stride),
             c->chunk_bytes,
             stream));
      }
    } else {
      CU(Fail,
         cuMemcpyDtoDAsync((CUdeviceptr)d_output,
                           (CUdeviceptr)d_input,
                           n * c->chunk_bytes,
                           stream));
    }
    return 0;
  }

  // Fill pointer arrays
  {
    unsigned blocks = (unsigned)((n + 255) / 256);
    fill_ptrs_kernel<<<blocks, 256, 0, cuda_stream>>>(c->d_ptrs,
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
                                           c->chunk_bytes,
                                           n,
                                           c->d_temp,
                                           c->temp_bytes,
                                           comp_ptrs,
                                           c->d_comp_sizes,
                                           nvcompBatchedLZ4CompressDefaultOpts,
                                           NULL,
                                           cuda_stream));
      break;

    case CODEC_ZSTD:
      NVCOMP(
        Fail,
        nvcompBatchedZstdCompressAsync(uncomp_ptrs,
                                       c->d_uncomp_sizes,
                                       c->chunk_bytes,
                                       n,
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

  return 0;

Fail:
  return 1;
}
