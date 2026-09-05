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

__global__ static void
fill_block_ptrs_kernel(void** ptrs,
                       size_t* sizes,
                       const char* input,
                       size_t input_stride,
                       size_t input_block_stride,
                       char* output,
                       size_t output_stride,
                       size_t chunk_bytes,
                       size_t block_bytes,
                       size_t blocks_per_chunk,
                       size_t count)
{
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const size_t chunk = i / blocks_per_chunk;
  const size_t offset = (i % blocks_per_chunk) * block_bytes;
  const size_t remaining = chunk_bytes - offset;
  ptrs[i] = (void*)(input + chunk * input_stride +
                    (i % blocks_per_chunk) * input_block_stride);
  ptrs[count + i] = output + i * output_stride;
  sizes[i] = remaining < block_bytes ? remaining : block_bytes;
}

static size_t
blosc_block_bytes(struct codec_config config, size_t chunk_bytes)
{
  return chunk_bytes < config.blosc_block_bytes ? chunk_bytes
                                                : config.blosc_block_bytes;
}

static size_t
blosc_block_count(struct codec_config config, size_t chunk_bytes)
{
  return chunk_bytes / config.blosc_block_bytes +
         (chunk_bytes % config.blosc_block_bytes != 0);
}

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

extern "C" size_t
codec_max_output_size(enum compression_codec type, size_t chunk_bytes)
{
  if (codec_is_blosc(type)) {
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
  const size_t bound = codec_max_output_size(type, chunk_bytes);
  if (alignment == 0 || bound == 0 || bound > SIZE_MAX - (alignment - 1))
    return 0;
  return align_up(bound, alignment);
}

static size_t
blosc_output_stride(struct codec_config config, size_t chunk_bytes)
{
  const size_t alignment = codec_output_alignment(config.id);
  const size_t bound =
    nvcomp_max_output_size(config.id, blosc_block_bytes(config, chunk_bytes));
  if (!alignment || !bound || bound > SIZE_MAX - (alignment - 1))
    return 0;
  return align_up(bound, alignment);
}

// Allocation sizes and capacity geometry are calculated together. Allocation
// and estimation consume this same result; codec_bind changes only active
// sizes.
struct codec_layout
{
  size_t block_bytes, blocks_per_chunk, inputs;
  size_t block_output_stride, block_input_stride;
  size_t max_output_size, output_stride;
  size_t comp_sizes, uncomp_sizes, ptrs;
  size_t block_input, block_sizes, block_offsets, block_data;
  size_t temp, total;
};

static int
allocation_size(size_t* bytes, size_t count, size_t stride, size_t* total)
{
  if (stride && count > (SIZE_MAX - *total) / stride)
    return 1;
  *bytes = count * stride;
  *total += *bytes;
  return 0;
}

static int
codec_layout_compute(struct codec_layout* l,
                     struct codec_config config,
                     size_t typesize,
                     size_t chunk_bytes,
                     size_t batch_size,
                     int reserve_shuffle)
{
  memset(l, 0, sizeof(*l));
  if (!batch_size || !chunk_bytes || batch_size > SIZE_MAX / chunk_bytes ||
      codec_validate_gpu(config, typesize, chunk_bytes))
    return 1;
  l->inputs = batch_size;
  if (codec_is_blosc(config.id)) {
    l->block_bytes = blosc_block_bytes(config, chunk_bytes);
    l->blocks_per_chunk = blosc_block_count(config, chunk_bytes);
    if (batch_size > INT_MAX / l->blocks_per_chunk)
      return 1;
    l->inputs *= l->blocks_per_chunk;
    l->block_output_stride = blosc_output_stride(config, chunk_bytes);
    if (!l->block_output_stride)
      return 1;
    const size_t alignment = codec_alignment(config.id);
    if (reserve_shuffle ||
        (l->blocks_per_chunk > 1 && l->block_bytes % alignment != 0))
      l->block_input_stride = align_up(l->block_bytes, alignment);
  }
  l->max_output_size = codec_max_output_size(config.id, chunk_bytes);
  l->output_stride = codec_output_stride(config.id, chunk_bytes);
  if (!l->max_output_size || !l->output_stride)
    return 1;

  switch (nvcomp_codec(config.id)) {
    case CODEC_NONE:
      break;
    case CODEC_LZ4_NON_STANDARD:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressGetTempSizeAsync(
               l->inputs,
               l->block_bytes ? l->block_bytes : chunk_bytes,
               nvcompBatchedLZ4CompressDefaultOpts,
               &l->temp,
               batch_size * chunk_bytes));
      break;
    case CODEC_ZSTD:
      NVCOMP(Fail,
             nvcompBatchedZstdCompressGetTempSizeAsync(
               l->inputs,
               l->block_bytes ? l->block_bytes : chunk_bytes,
               nvcompBatchedZstdCompressDefaultOpts,
               &l->temp,
               batch_size * chunk_bytes));
      break;
    default:
      return 1;
  }

  l->total = l->temp;
  if (allocation_size(&l->comp_sizes, batch_size, sizeof(size_t), &l->total) ||
      allocation_size(&l->uncomp_sizes, l->inputs, sizeof(size_t), &l->total) ||
      allocation_size(&l->ptrs,
                      config.id != CODEC_NONE ? l->inputs : 0,
                      2 * sizeof(void*),
                      &l->total) ||
      allocation_size(
        &l->block_input, l->inputs, l->block_input_stride, &l->total) ||
      allocation_size(&l->block_sizes,
                      l->blocks_per_chunk ? l->inputs : 0,
                      sizeof(size_t),
                      &l->total) ||
      allocation_size(&l->block_offsets,
                      l->blocks_per_chunk ? l->inputs : 0,
                      sizeof(size_t),
                      &l->total) ||
      allocation_size(
        &l->block_data, l->inputs, l->block_output_stride, &l->total))
    return 1;
  return 0;
Fail:
  return 1;
}

extern "C" size_t
codec_device_bytes(struct codec_config config,
                   size_t chunk_bytes,
                   size_t batch_size,
                   int reserve_shuffle)
{
  struct codec_layout layout;
  return codec_layout_compute(
           &layout, config, 1, chunk_bytes, batch_size, reserve_shuffle)
           ? 0
           : layout.total;
}

extern "C" size_t
codec_temp_bytes(struct codec_config config,
                 size_t chunk_bytes,
                 size_t batch_size)
{
  struct codec_layout layout;
  return codec_layout_compute(&layout, config, 1, chunk_bytes, batch_size, 0)
           ? 0
           : layout.temp;
}

extern "C" int
codec_init(struct codec* c,
           enum compression_codec type,
           size_t chunk_bytes,
           size_t batch_size)
{
  struct codec_config config = {
    .id = type,
    .level = (uint8_t)(type == CODEC_LZ4_NON_STANDARD ? 1 : 0),
    .shuffle = CODEC_SHUFFLE_NONE,
    .blosc_block_bytes = 0,
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
  if (codec_config_validate_blosc(config))
    return 1;
  if (typesize == 0 || typesize > 255) {
    log_error("GPU blosc typesize must be 1..255 (got %zu)", typesize);
    return 1;
  }
  if (codec_max_output_size(config.id, chunk_bytes) == 0) {
    log_error("GPU blosc chunk size %zu exceeds the Blosc frame limit",
              chunk_bytes);
    return 1;
  }
  const size_t block_bytes = blosc_block_bytes(config, chunk_bytes);
  const size_t nvcomp_limit = config.id == CODEC_BLOSC_LZ4
                                ? nvcompLZ4CompressionMaxAllowedChunkSize
                                : nvcompZstdCompressionMaxAllowedChunkSize;
  if (block_bytes > nvcomp_limit ||
      block_bytes > ((size_t)INT_MAX - 255 * 4) / 3) {
    log_error("GPU blosc block size %zu exceeds the codec limit", block_bytes);
    return 1;
  }
  if (codec_output_stride(config.id, chunk_bytes) == 0) {
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
                  int reserve_shuffle)
{
  struct codec_layout layout;
  memset(c, 0, sizeof(*c));
  CHECK(
    Fail,
    codec_layout_compute(
      &layout, config, typesize, chunk_bytes, batch_size, reserve_shuffle) ==
      0);
  c->type = config.id;
  c->config = config;
  c->typesize = typesize;
  c->chunk_bytes = chunk_bytes;
  c->chunk_capacity = chunk_bytes;
  c->batch_size = batch_size;
  c->block_bytes = layout.block_bytes;
  c->blocks_per_chunk = layout.blocks_per_chunk;
  c->block_capacity = layout.blocks_per_chunk ? layout.inputs : 0;
  c->block_output_stride = layout.block_output_stride;
  c->block_input_stride = layout.block_input_stride;
  c->max_output_size = layout.max_output_size;
  c->output_stride = layout.output_stride;
  c->temp_bytes = layout.temp;

  CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_comp_sizes, layout.comp_sizes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_uncomp_sizes, layout.uncomp_sizes));

  if (!codec_is_blosc(config.id)) {
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

  if (config.id != CODEC_NONE) {
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_ptrs, layout.ptrs));
  }

  if (codec_is_blosc(config.id)) {
    if (c->block_input_stride)
      CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_block_input, layout.block_input));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_block_sizes, layout.block_sizes));
    CU(Fail,
       cuMemAlloc((CUdeviceptr*)&c->d_block_offsets, layout.block_offsets));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_block_data, layout.block_data));
  }

  if (c->temp_bytes > 0) {
    CU(Fail, cuMemAlloc((CUdeviceptr*)&c->d_temp, c->temp_bytes));
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
  if (codec_is_blosc(c->type) &&
      config.blosc_block_bytes != c->config.blosc_block_bytes) {
    log_error("GPU blosc_block_bytes cannot change on a shared codec instance");
    goto Fail;
  }
  if (codec_is_blosc(c->type) && config.shuffle != CODEC_SHUFFLE_NONE &&
      !c->d_block_input) {
    log_error("GPU blosc block preparation scratch was not reserved");
    goto Fail;
  }
  if (chunk_bytes == c->chunk_bytes && typesize == c->typesize &&
      config.id == c->config.id && config.level == c->config.level &&
      config.shuffle == c->config.shuffle)
    return 0;

  if (chunk_bytes != c->chunk_bytes && !codec_is_blosc(c->type)) {
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
  if (codec_is_blosc(c->type)) {
    c->block_bytes = blosc_block_bytes(config, chunk_bytes);
    c->blocks_per_chunk = blosc_block_count(config, chunk_bytes);
  }
  return 0;

Fail:
  return 1;
}

extern "C" void
codec_free(struct codec* c)
{
  cu_mem_free((CUdeviceptr)c->d_comp_sizes);
  cu_mem_free((CUdeviceptr)c->d_uncomp_sizes);
  cu_mem_free((CUdeviceptr)c->d_ptrs);
  cu_mem_free((CUdeviceptr)c->d_temp);
  cu_mem_free((CUdeviceptr)c->d_block_data);
  cu_mem_free((CUdeviceptr)c->d_block_input);
  cu_mem_free((CUdeviceptr)c->d_block_sizes);
  cu_mem_free((CUdeviceptr)c->d_block_offsets);
  c->d_comp_sizes = NULL;
  c->d_uncomp_sizes = NULL;
  c->d_ptrs = NULL;
  c->d_temp = NULL;
  c->d_block_data = NULL;
  c->d_block_input = NULL;
  c->d_block_sizes = NULL;
  c->d_block_offsets = NULL;
}

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
  void* const* comp_ptrs = NULL;
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int is_blosc = codec_is_blosc(c->type);
  const int force_copy =
    is_blosc &&
    (c->config.level == 0 || c->chunk_bytes < GPU_BLOSC_MIN_COMPRESS_BYTES);
  size_t inputs = 0;
  const size_t input_bytes = is_blosc ? c->block_bytes : c->chunk_bytes;
  size_t* output_sizes = is_blosc ? c->d_block_sizes : c->d_comp_sizes;
  const struct gpu_blosc_frame_layout frame = {
    c->type, c->config.shuffle, c->typesize, c->chunk_bytes, c->block_bytes
  };
  const struct gpu_blosc_input original = { d_input, input_stride };
  const struct gpu_blosc_blocks blocks = { uncomp_ptrs,
                                           c->d_block_data,
                                           c->block_output_stride,
                                           c->d_block_sizes,
                                           c->d_block_offsets };
  const struct gpu_blosc_output encoded = { d_output,
                                            c->output_stride,
                                            c->d_comp_sizes };
  struct gpu_blosc_input input = original;
  size_t input_block_stride = c->block_bytes;

  CHECK(Fail, n > 0 && n <= c->batch_size);
  CHECK(Fail, input_stride >= c->chunk_bytes);

  if (c->type == CODEC_NONE) {
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
          gpu_blosc_pack_async(
            frame, original, blocks, encoded, n, 1, stream) == 0);
    return 0;
  }

  // Filtering and alignment share one prepared-block buffer. Already aligned,
  // unfiltered blocks can be handed directly to nvCOMP and the frame packer.
  if (is_blosc &&
      ((c->config.shuffle != CODEC_SHUFFLE_NONE &&
        !(c->config.shuffle == CODEC_SHUFFLE_BYTE && c->typesize == 1)) ||
       (c->blocks_per_chunk > 1 &&
        c->block_bytes % codec_alignment(c->type)))) {
    CHECK(Fail, c->d_block_input);
    CHECK(
      Fail,
      gpu_blosc_prepare_blocks_async(
        frame, original, c->d_block_input, c->block_input_stride, n, stream) ==
        0);
    input_block_stride = c->block_input_stride;
    input =
      (struct gpu_blosc_input){ c->d_block_input,
                                c->blocks_per_chunk * input_block_stride };
  }

  inputs = is_blosc ? n * c->blocks_per_chunk : n;
  comp_ptrs = (void* const*)(c->d_ptrs + inputs);
  if (is_blosc) {
    const unsigned blocks = (unsigned)((inputs + 255) / 256);
    CUDA_LAUNCH_OR(Fail,
                   fill_block_ptrs_kernel<<<blocks, 256, 0, cuda_stream>>>(
                     c->d_ptrs,
                     c->d_uncomp_sizes,
                     (const char*)input.data,
                     input.stride,
                     input_block_stride,
                     (char*)c->d_block_data,
                     c->block_output_stride,
                     c->chunk_bytes,
                     c->block_bytes,
                     c->blocks_per_chunk,
                     inputs));
  } else {
    unsigned blocks = (unsigned)((n + 255) / 256);
    CUDA_LAUNCH_OR(
      Fail,
      fill_ptrs_kernel<<<blocks, 256, 0, cuda_stream>>>(c->d_ptrs,
                                                        (const char*)d_input,
                                                        input_stride,
                                                        (char*)d_output,
                                                        c->output_stride,
                                                        n));
  }

  switch (nvcomp_codec(c->type)) {
    case CODEC_LZ4_NON_STANDARD:
      NVCOMP(Fail,
             nvcompBatchedLZ4CompressAsync(uncomp_ptrs,
                                           c->d_uncomp_sizes,
                                           input_bytes,
                                           inputs,
                                           c->d_temp,
                                           c->temp_bytes,
                                           comp_ptrs,
                                           output_sizes,
                                           nvcompBatchedLZ4CompressDefaultOpts,
                                           NULL,
                                           cuda_stream));
      break;

    case CODEC_ZSTD:
      NVCOMP(
        Fail,
        nvcompBatchedZstdCompressAsync(uncomp_ptrs,
                                       c->d_uncomp_sizes,
                                       input_bytes,
                                       inputs,
                                       c->d_temp,
                                       c->temp_bytes,
                                       comp_ptrs,
                                       output_sizes,
                                       nvcompBatchedZstdCompressDefaultOpts,
                                       NULL,
                                       cuda_stream));
      break;

    default:
      goto Fail;
  }

  if (is_blosc) {
    CHECK(Fail,
          gpu_blosc_pack_async(
            frame, original, blocks, encoded, n, 0, stream) == 0);
  }

  return 0;

Fail:
  return 1;
}
