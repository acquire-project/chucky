#include "gpu/blosc.frame.h"
#include "gpu/blosc.shuffle.h"
#include "gpu/compress.h"
#include "gpu/prelude.cuda.h"
#include "log/log.h"
#include "util/prelude.h"
#include <limits.h>
#include <nvcomp/lz4.h>
#include <nvcomp/shared_types.h>
#include <nvcomp/zstd.h>
#include <string.h>

static enum compression_codec
nvcomp_codec(enum compression_codec type)
{
  switch (type) {
    case CODEC_BLOSC_LZ4:
      return CODEC_LZ4_NON_STANDARD;
    case CODEC_BLOSC_ZSTD:
      return CODEC_ZSTD;
    default:
      return type;
  }
}

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

__global__ void
fill_sizes_kernel(size_t* d_sizes, size_t value, size_t count)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    d_sizes[i] = value;
}

// --- codec_alignment ---

extern "C" size_t
codec_alignment(enum compression_codec type)
{
  type = nvcomp_codec(type);
  switch (type) {
    case CODEC_LZ4_NON_STANDARD:
      return nvcompLZ4RequiredCompressionAlignment;
    case CODEC_ZSTD:
      return nvcompZstdRequiredCompressionAlignment;
    default:
      return 1;
  }
}

// --- codec_output_alignment ---

// nvcomp's named alignment constants are the strictest across all its buffers,
// set by one this path never allocates; compressed chunks need only the
// alignment nvcomp reports for them.
extern "C" size_t
codec_output_alignment(enum compression_codec type)
{
  type = nvcomp_codec(type);
  nvcompAlignmentRequirements_t written = {}, read = {};
  switch (type) {
    case CODEC_LZ4_NON_STANDARD:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetRequiredAlignments(
               nvcompBatchedLZ4CompressDefaultOpts, &written));
      NVCOMP(Fail,
             nvcompBatchedLZ4DecompressGetRequiredAlignments(
               nvcompBatchedLZ4DecompressDefaultOpts, &read));
      break;
    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetRequiredAlignments(
               nvcompBatchedZstdCompressDefaultOpts, &written));
      NVCOMP(Fail,
             nvcompBatchedZstdDecompressGetRequiredAlignments(
               nvcompBatchedZstdDecompressDefaultOpts, &read));
      break;
    default:
      return 1;
  }
  return written.output > read.input ? written.output : read.input;
Fail:
  return 0;
}

// --- codec_max_output_size ---

extern "C" size_t
codec_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  if (codec_is_blosc(type)) {
    // C-Blosc stores these fields as signed 32-bit values and limits the
    // source to INT_MAX - its 16-byte maximum overhead.
    if (chunk_bytes == 0 ||
        chunk_bytes > (size_t)INT_MAX - GPU_BLOSC_HEADER_BYTES)
      return 0;
    return chunk_bytes + GPU_BLOSC_HEADER_BYTES;
  }

  size_t max_comp = 0;
  const size_t alignment = codec_output_alignment(type);
  CHECK(Fail, alignment > 0);

  switch (type) {
    case CODEC_NONE:
      max_comp = chunk_bytes;
      break;
    case CODEC_LZ4_NON_STANDARD:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedLZ4CompressDefaultOpts, &max_comp));
      break;
    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetMaxOutputChunkSize(
               chunk_bytes, nvcompBatchedZstdCompressDefaultOpts, &max_comp));
      break;
    default:
      goto Fail;
  }
  // nvcomp's own bound need not be a multiple of the alignment it asks for.
  return align_up(max_comp, alignment);
Fail:
  return 0;
}

static size_t
nvcomp_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  size_t max_comp = 0;
  switch (nvcomp_codec(type)) {
    case CODEC_LZ4_NON_STANDARD:
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
      return chunk_bytes;
  }
Fail:
  return 0;
}

extern "C" size_t
codec_output_stride(enum compression_codec type, size_t chunk_bytes)
{
  if (!codec_is_blosc(type))
    return codec_max_output_size(type, chunk_bytes);

  const size_t alignment = codec_output_alignment(type);
  const size_t payload_bound = nvcomp_max_output_size(type, chunk_bytes);
  if (alignment == 0 || payload_bound == 0 ||
      GPU_BLOSC_PAYLOAD_OFFSET % alignment != 0 ||
      payload_bound > SIZE_MAX - GPU_BLOSC_PAYLOAD_OFFSET)
    return 0;
  const size_t framed = GPU_BLOSC_PAYLOAD_OFFSET + payload_bound;
  if (framed > SIZE_MAX - (alignment - 1))
    return 0;
  return align_up(framed, alignment);
}

// --- codec_temp_bytes ---

extern "C" size_t
codec_temp_bytes(enum compression_codec type,
                 size_t chunk_bytes,
                 size_t batch_size)
{
  type = nvcomp_codec(type);
  size_t temp = 0;
  switch (type) {
    case CODEC_NONE:
      return 0;
    case CODEC_LZ4_NON_STANDARD:
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

// --- codec_device_bytes ---

// Device bytes codec_init allocates for this configuration. Must mirror the
// allocations below exactly — tile_stream_gpu_memory_estimate sums this.
extern "C" size_t
codec_device_bytes(enum compression_codec type,
                   size_t chunk_bytes,
                   size_t batch_size,
                   int reserve_byte_shuffle)
{
  if (batch_size == 0 || batch_size > SIZE_MAX / sizeof(size_t))
    return 0;
  size_t bytes = batch_size * sizeof(size_t); // d_comp_sizes
  if (bytes > SIZE_MAX - batch_size * sizeof(size_t))
    return 0;
  bytes += batch_size * sizeof(size_t); // d_uncomp_sizes
  if (type != CODEC_NONE) {
    if (batch_size > SIZE_MAX / (2 * sizeof(void*)) ||
        2 * batch_size * sizeof(void*) > SIZE_MAX - bytes)
      return 0;
    bytes += 2 * batch_size * sizeof(void*); // d_ptrs
  }
  const size_t temp = codec_temp_bytes(type, chunk_bytes, batch_size);
  if (temp > SIZE_MAX - bytes)
    return 0;
  bytes += temp; // d_temp
  if (codec_is_blosc(type) && reserve_byte_shuffle) {
    const size_t alignment = codec_alignment(type);
    if (alignment == 0 || chunk_bytes > SIZE_MAX - (alignment - 1))
      return 0;
    const size_t shuffle_stride = align_up(chunk_bytes, alignment);
    if (shuffle_stride > SIZE_MAX / batch_size ||
        shuffle_stride * batch_size > SIZE_MAX - bytes)
      return 0;
    bytes += shuffle_stride * batch_size; // d_shuffle
  }
  return bytes;
}

// --- codec_init ---

extern "C" int
codec_init(struct codec* c,
           enum compression_codec type,
           size_t chunk_bytes,
           size_t batch_size)
{
  struct codec_config config = {
    .id = type,
    .level = (uint8_t)(type == CODEC_LZ4_NON_STANDARD ? 1 : 0),
    .shuffle = CODEC_SHUFFLE_NONE
  };
  return codec_init_config(c, config, 1, chunk_bytes, batch_size, 0);
}

extern "C" int
codec_validate_gpu(struct codec_config config,
                   size_t typesize,
                   size_t chunk_bytes)
{
  if (!codec_is_gpu_supported(config.id)) {
    log_error("codec %d is not supported on GPU", (int)config.id);
    return 1;
  }
  if (!codec_is_blosc(config.id))
    return 0;
  if (config.level > 9) {
    log_error("blosc level must be 0..9 (got %u)", config.level);
    return 1;
  }
  if (config.shuffle == CODEC_SHUFFLE_BIT) {
    log_error("GPU blosc bitshuffle is not supported");
    return 1;
  }
  if (config.shuffle != CODEC_SHUFFLE_NONE &&
      config.shuffle != CODEC_SHUFFLE_BYTE) {
    log_error("invalid GPU blosc shuffle mode %d", (int)config.shuffle);
    return 1;
  }
  if (typesize == 0 || typesize > 255) {
    log_error("GPU blosc typesize must be 1..255 (got %zu)", typesize);
    return 1;
  }
  // A single block must fit C-Blosc's decompressor scratch-size contract.
  const size_t max_block = ((size_t)INT_MAX - 255 * 4) / 3;
  if (chunk_bytes == 0 || chunk_bytes > max_block) {
    log_error("GPU blosc single-block input %zu exceeds limit %zu",
              chunk_bytes,
              max_block);
    return 1;
  }
  if (codec_max_output_size(config.id, chunk_bytes) == 0 ||
      codec_output_stride(config.id, chunk_bytes) == 0) {
    log_error("GPU blosc does not support %zu-byte chunks", chunk_bytes);
    return 1;
  }
  return 0;
}

extern "C" int
codec_init_config(struct codec* c,
                  struct codec_config config,
                  size_t typesize,
                  size_t chunk_bytes,
                  size_t batch_size,
                  int reserve_byte_shuffle)
{
  memset(c, 0, sizeof(*c));
  c->type = config.id;
  c->config = config;
  c->typesize = typesize;
  c->chunk_bytes = chunk_bytes;
  c->chunk_capacity = chunk_bytes;
  c->batch_size = batch_size;

  if (batch_size == 0 || codec_validate_gpu(config, typesize, chunk_bytes))
    goto Fail;

  c->max_output_size = codec_max_output_size(config.id, chunk_bytes);
  CHECK(Fail, c->max_output_size > 0);
  c->output_stride = codec_output_stride(config.id, chunk_bytes);
  CHECK(Fail, c->output_stride > 0);

  c->temp_bytes = codec_temp_bytes(config.id, chunk_bytes, batch_size);

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
  if (config.id == CODEC_NONE) {
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
  if (config.id != CODEC_NONE) {
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&c->d_ptrs, 2 * batch_size * sizeof(void*)));
  }

  // Temp workspace
  if (c->temp_bytes > 0) {
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_temp, c->temp_bytes));
  }

  if (codec_is_blosc(config.id) && reserve_byte_shuffle) {
    c->shuffle_stride = align_up(chunk_bytes, codec_alignment(config.id));
    CHECK_MUL_OVERFLOW(Fail, batch_size, c->shuffle_stride, SIZE_MAX);
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&c->d_shuffle, batch_size * c->shuffle_stride));
    c->has_shuffle_scratch = 1;
  }

  return 0;

Fail:
  codec_free(c);
  return 1;
}

extern "C" int
codec_set_chunk_bytes(struct codec* c, size_t chunk_bytes, CUstream stream)
{
  return codec_bind(c, c->config, c->typesize, chunk_bytes, stream);
}

extern "C" int
codec_bind(struct codec* c,
           struct codec_config config,
           size_t typesize,
           size_t chunk_bytes,
           CUstream stream)
{
  const unsigned block = 256;
  unsigned grid = 0;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  CHECK(Fail,
        c && config.id == c->type && chunk_bytes > 0 &&
          chunk_bytes <= c->chunk_capacity);
  CHECK(Fail, codec_validate_gpu(config, typesize, chunk_bytes) == 0);
  if (config.shuffle == CODEC_SHUFFLE_BYTE && !c->has_shuffle_scratch) {
    log_error("GPU blosc byte shuffle scratch was not reserved");
    goto Fail;
  }
  if (chunk_bytes == c->chunk_bytes && typesize == c->typesize &&
      config.id == c->config.id && config.level == c->config.level &&
      config.shuffle == c->config.shuffle)
    return 0;

  if (chunk_bytes != c->chunk_bytes) {
    grid = (unsigned)((c->batch_size + block - 1) / block);
    CUDA_LAUNCH_OR(Fail,
                   fill_sizes_kernel<<<grid, block, 0, cuda_stream>>>(
                     c->d_uncomp_sizes, chunk_bytes, c->batch_size));
    if (c->type == CODEC_NONE)
      CUDA_LAUNCH_OR(Fail,
                     fill_sizes_kernel<<<grid, block, 0, cuda_stream>>>(
                       c->d_comp_sizes, chunk_bytes, c->batch_size));
  }
  c->config = config;
  c->typesize = typesize;
  c->chunk_bytes = chunk_bytes;
  return 0;

Fail:
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
  cu_mem_free((CUdeviceptr)c->d_shuffle);
  c->d_comp_sizes = NULL;
  c->d_uncomp_sizes = NULL;
  c->d_ptrs = NULL;
  c->d_temp = NULL;
  c->d_shuffle = NULL;
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
  const int is_blosc = codec_is_blosc(c->type);
  const int force_copy =
    is_blosc &&
    (c->config.level == 0 || c->chunk_bytes < GPU_BLOSC_MIN_COMPRESS_BYTES);
  const void* nvcomp_input = d_input;
  size_t nvcomp_input_stride = input_stride;

  CHECK(Fail, n > 0 && n <= c->batch_size);

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

  if (force_copy) {
    CHECK(Fail,
          gpu_blosc_finalize_async(c->type,
                                   c->config.shuffle,
                                   c->typesize,
                                   c->chunk_bytes,
                                   d_input,
                                   input_stride,
                                   d_output,
                                   c->output_stride,
                                   c->d_comp_sizes,
                                   n,
                                   1,
                                   stream) == 0);
    return 0;
  }

  if (is_blosc && c->config.shuffle == CODEC_SHUFFLE_BYTE && c->typesize > 1) {
    CHECK_MUL_OVERFLOW(Fail, n, c->chunk_bytes, SIZE_MAX);
    CHECK(Fail,
          gpu_blosc_shuffle_async(d_input,
                                  input_stride,
                                  c->d_shuffle,
                                  c->shuffle_stride,
                                  c->chunk_bytes,
                                  c->typesize,
                                  n,
                                  stream) == 0);
    nvcomp_input = c->d_shuffle;
    nvcomp_input_stride = c->shuffle_stride;
  }

  // Fill pointer arrays
  {
    unsigned blocks = (unsigned)((n + 255) / 256);
    CUDA_LAUNCH_OR(
      Fail,
      fill_ptrs_kernel<<<blocks, 256, 0, cuda_stream>>>(
        c->d_ptrs,
        (const char*)nvcomp_input,
        nvcomp_input_stride,
        (char*)d_output + (is_blosc ? GPU_BLOSC_PAYLOAD_OFFSET : 0),
        c->output_stride,
        n));
  }

  switch (nvcomp_codec(c->type)) {
    case CODEC_LZ4_NON_STANDARD:
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

  if (is_blosc) {
    CHECK(Fail,
          gpu_blosc_finalize_async(c->type,
                                   c->config.shuffle,
                                   c->typesize,
                                   c->chunk_bytes,
                                   d_input,
                                   input_stride,
                                   d_output,
                                   c->output_stride,
                                   c->d_comp_sizes,
                                   n,
                                   0,
                                   stream) == 0);
  }

  return 0;

Fail:
  return 1;
}
