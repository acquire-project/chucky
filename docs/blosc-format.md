# Blosc Encoding Conformance Profile

Profile revision: 1\
Date: 2026-09-05

## Abstract

This document records the C-Blosc 1.x binary format to which the CPU and GPU
encoders in this repository conform. It specifies the supported LZ4 and
Zstandard representations, the GPU encoder's narrower output profile, and the
requirements for reading output from either backend. It defines no new Blosc
format or compressor identifiers.

## Relationship to C-Blosc

The baseline is the upstream
[Blosc Chunk Format description shipped with C-Blosc 1.21.6][c-blosc-format],
at commit `616f4b7343a8479f7e71dd3d7025bd92c9a6bbd0` (2024-06-24).
The [implementation at that same revision][c-blosc-reference]
resolves details omitted or misstated in that description. In particular:

- The upstream description labels header sizes as unsigned 32-bit integers;
  this profile restricts them to the positive signed 32-bit range used by
  the reference implementation.
- Its description places the origin of `bstarts` after the offset table.
  The reference implementation instead uses offsets from the beginning of
  the encoded chunk, as specified in Section 5 here.
- Sections 4–6 make the verbatim forms, split streams, and filter ordering
  explicit, including details not fully specified in the upstream description.

The numbered requirements below are the conformance target for this profile
revision. Later upstream documentation or library releases do not implicitly
revise them. A change to the supported representation requires an explicit
profile revision; implementation changes that preserve it do not.

In this document, **the encoder** means the writer in this repository
(currently named Chucky), not every C-Blosc encoder. **GPU profile** identifies
the stricter requirements on its GPU output. A reader supporting both backends
also needs the CPU representations in Section 5.3. This distinction describes
stored bytes, not an execution strategy or performance guarantee.

## 1. Scope and Conventions

Capitalized requirement terms (MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT,
and MAY) carry the meanings assigned by BCP 14
([RFC 2119][rfc2119] and
[RFC 8174][rfc8174]). Lowercase usage is ordinary
prose.

An **octet** is eight bits. `u8` denotes one unsigned octet. `i32le` denotes
a four-octet, little-endian signed integer. Sizes and offsets in conforming
output are nonnegative; readers MUST reject negative values where a size or
offset is required. Offsets are measured in octets from the first header octet.
Bit zero is the least-significant bit of an octet.

A **chunk** is one complete uncompressed Zarr inner chunk, including any
edge padding supplied by the array encoder. A **block** is a contiguous portion
of that chunk. A **stream** is one independently encoded payload within a block.

One Blosc object encodes one chunk. This document calls that object an
**encoded chunk**; the source also calls it a frame. It is NOT a Blosc2
super-chunk, contiguous frame, or extended-header chunk. The header version
value `2` below is the C-Blosc 1.x format version, not a request for Blosc2.

Only the legacy, 16-octet-header representation is in scope. The evolving
[Blosc/Blosc2 chunk description][blosc2-format]
also covers extended headers, additional filters, dictionaries, special values,
and variable-length blocks; those extensions are not part of this profile.

## 2. Encoded Chunk Layout

Every encoded chunk starts with the following 16-octet header:

```text
Octet     0          1          2          3
       +----------+----------+----------+----------+
       | version  |versionlz |  flags   | typesize |
       +----------+----------+----------+----------+
       |               nbytes (i32le)              |  4..7
       +-------------------------------------------+
       |              blocksize (i32le)            |  8..11
       +-------------------------------------------+
       |               cbytes (i32le)              | 12..15
       +-------------------------------------------+
```

| Field | Type | Meaning in the GPU profile |
|---|---|---|
| `version` | u8 | MUST be 2 |
| `versionlz` | u8 | MUST be 1 for both supported codecs |
| `flags` | u8 | Filter, storage mode, split mode, and compressor; Section 3 |
| `typesize` | u8 | Element width in octets, 1 through 255 |
| `nbytes` | i32le | Uncompressed chunk length, excluding this header |
| `blocksize` | i32le | Actual uncompressed full-block length |
| `cbytes` | i32le | Complete encoded length, including this header |

The GPU profile requires `1 <= nbytes <= 2147483631`,
`1 <= blocksize <= nbytes`, and `16 < cbytes <= nbytes + 16`.
The stored block size MUST be used for decoding. It need not equal the requested
block size in the application's configuration or Zarr metadata.

No magic string, chunk dimensions, compression level, or per-chunk checksum
is present in this header. Bytes beyond `cbytes` do not belong to the encoded
chunk. Device-slot padding and outer shard padding MUST NOT be interpreted as
part of this representation.

## 3. Flags

| Bits | Mask | Meaning |
|---|---|---|
| 0 | 0x01 | Byte shuffle selected |
| 1 | 0x02 | `MEMCPYED`: whole-chunk verbatim representation |
| 2 | 0x04 | Bitshuffle selected |
| 3 | 0x08 | Reserved in this profile; MUST be zero |
| 4 | 0x10 | `DONT_SPLIT`: one stream per block |
| 5..7 | 0xe0 | Compressor format identifier |

The compressor identifier is `(flags >> 5) & 7`. The encoder writes `1` for LZ4
and `4` for Zstandard. These are format identifiers, not values of the API's
`enum compression_codec`.

The GPU encoder MUST set `DONT_SPLIT`. It MUST NOT set both filter bits.
The filter bits describe the configured filter; they do not imply that bytes
were transformed when that filter is an identity operation. In particular,
`MEMCPYED` takes precedence over both filter bits.

Without `MEMCPYED`, the GPU flags are:

| Filter | LZ4 | Zstandard |
|---|---:|---:|
| None | 0x30 | 0x90 |
| Byte shuffle | 0x31 | 0x91 |
| Bitshuffle | 0x34 | 0x94 |

The corresponding verbatim form adds `0x02`.

## 4. Whole-Chunk Verbatim Representation

When `MEMCPYED` is set, the header MUST be followed immediately by exactly
`nbytes` octets of original, unfiltered chunk data. `cbytes` MUST equal
`16 + nbytes`. There is no block-offset table and no stream-length prefix.

```text
       +-------------------+-------------------------------+
       | header: 16 octets | original chunk: nbytes octets |
       +-------------------+-------------------------------+
```

A decoder MUST copy these bytes without decompression or inverse filtering,
even if a filter flag is set. The block size has no effect on this copy.

The GPU encoder uses this form for level 0, for chunks shorter than 128 octets,
and whenever the complete block representation would not be strictly smaller
than `nbytes + 16`. Compression-level metadata is not needed to recognize it.

## 5. Block Representation

When `MEMCPYED` is clear, define:

```text
N = ceil(nbytes / blocksize)
U[i] = min(blocksize, nbytes - i * blocksize),  0 <= i < N
```

The header MUST be followed by `N` four-octet `i32le` block offsets,
called `bstarts`. Entry `i` locates the encoded block for uncompressed chunk
range `[i * blocksize, i * blocksize + U[i])`.

```text
       +--------+-----------------+---------+---------+-----+
       | header | bstarts: 4*N    | block 0 | block 1 | ... |
       +--------+-----------------+---------+---------+-----+
         16 B       4*N B          variable lengths
```

### 5.1. GPU Profile: One Stream per Block

Each GPU block has exactly one stream:

```text
       +----------------+--------------------------------+
       | csize: i32le   | payload: csize octets          |
       +----------------+--------------------------------+
```

For block `i`, `1 <= csize <= U[i]` MUST hold.

- If `csize < U[i]`, the payload is the selected codec's encoding of the
  filtered block. Decoding it MUST produce exactly `U[i]` octets.
- If `csize == U[i]`, the payload is the uncompressed, **filtered** block.
  A decoder MUST copy it and then apply the inverse filter.

This per-block raw representation differs from whole-chunk `MEMCPYED`, whose
bytes are unfiltered. A single chunk MAY contain compressed and raw blocks.
Zero and negative stream-length special encodings are not part of this profile.

GPU blocks MUST be tightly packed in increasing logical block order:

```text
bstarts[0]   = 16 + 4*N
bstarts[i+1] = bstarts[i] + 4 + csize[i]
cbytes       = bstarts[N-1] + 4 + csize[N-1]
```

Neither payloads nor length fields have an alignment requirement. For example,
with two 256-octet blocks, an 11-octet first payload and a 256-octet raw second
payload, the offsets are 24 and 39 and the encoded length is 299. The second
length prefix begins at an unaligned address.

### 5.2. Codec Payloads

For LZ4, each compressed payload is an independent
[LZ4 block][lz4-block-format], not an
LZ4 frame. Its uncompressed length comes from `U[i]`; it MUST NOT require a
dictionary or a history from another block.

For Zstandard, each compressed payload is an independent Zstandard frame as
specified by [RFC 8878][rfc8878]. It MUST NOT
require an external dictionary or another Blosc block. There is no extra
nvCOMP container around either codec's payload.

### 5.3. CPU Output and Split Blocks

The CPU encoder conforms to the C-Blosc 1.x representation. It shares this
header and the verbatim representation, but a reader for **both** backends MUST NOT impose the
GPU-only `DONT_SPLIT` or tight physical ordering requirements on CPU chunks.

When `DONT_SPLIT` is clear, a full block is divided into `typesize` consecutive
streams of equal uncompressed length, `blocksize / typesize`. A shorter final
block remains one stream. Each stream has its own `csize` and payload; the
equal-length raw marker applies to that stream's uncompressed length.
The decoded streams are concatenated before the block is inverse-filtered.
The offset table, rather than assumed physical ordering, identifies blocks.

C-Blosc may adjust the requested block size for splitting or element alignment
and may choose split behavior based on codec and level. A CPU decoder MUST
follow the stored header and the
[pinned C-Blosc 1.x rules][c-blosc-reference],
not reconstruct those choices from application defaults. Byte-identical CPU
and GPU output is not required.

## 6. Filters

Filters act independently within each block before compression. They MUST NOT
carry state across block boundaries. In the following definitions, `S` is a
block of `U` original octets, `T = typesize`, `E = floor(U/T)`, and `F` is its
filtered representation. Bytes at offsets `E*T` through `U-1` are unchanged.

### 6.1. No Filter

`F[j] = S[j]` for every block byte.

### 6.2. Byte Shuffle

For `0 <= b < T` and `0 <= e < E`:

```text
F[b*E + e] = S[e*T + b]
```

For `T = 1` this is the identity. An inverse shuffle restores the original
element-byte ordering after decoding each block.

### 6.3. Bitshuffle

This profile uses the C-Blosc 1.x bitshuffle ordering. If `E` is not divisible
by eight, the entire block is unchanged. Otherwise, let `G = E/8`. For
`0 <= b < T`, `0 <= k < 8`, and `0 <= g < G`:

```text
F[(b*8 + k)*G + g] =
    sum((((S[(g*8 + e)*T + b] >> k) & 1) << e), e = 0..7)
```

Here `k = 0` selects the least-significant bit. The definition concerns stored
bytes; it does not reinterpret numeric values or change their byte order.

## 7. Zarr Binding and Configuration

The Zarr bytes-to-bytes codec name is `blosc`, with `cname` equal to `lz4` or
`zstd`. The encoder records `clevel`, `shuffle` (`noshuffle`, `shuffle`, or
`bitshuffle`), `typesize`, and the requested `blocksize` in codec configuration.
Outer shard indexes locate complete encoded chunks; their lengths exclude any
outer alignment padding.

The API requires an explicit `codec_config.blosc_block_bytes` in the range
128 through 715827542, including at level 0. Zero is invalid. GPU block sizes
are capped to the chunk length and are additionally subject to nvCOMP's input
size limit. The header records the actual block size, not that API limit.

Levels 1 through 9 on GPU select the same nvCOMP compression mode. CPU C-Blosc
honors the level. Neither reader needs a compression level to decode the bytes.

## 8. Decoder Validation and Security Considerations

A reader SHOULD enforce application-specific uncompressed size and resource
limits before allocating memory. A small encoded object may describe a much
larger uncompressed chunk.

A conforming GPU-profile validator MUST check at least:

1. Availability of the complete header, supported versions and flags, and valid
   positive sizes before arithmetic or allocation.
2. `cbytes` fits inside the supplied encoded input and the output capacity is
   at least `nbytes`.
3. The verbatim form has exactly the length required by Section 4.
4. For block encoding, the complete offset table fits within `cbytes`; every
   prefix and payload is in bounds; and Section 5.1's packing equalities hold.
5. Each decompressed block has its expected length. The complete output has
   exactly `nbytes` bytes.

All sums and products MUST be checked for overflow. Serialized integer reads
MUST support unaligned addresses. A validator for CPU output must instead
allow the split representation and physical ordering described in Section 5.3.

These are requirements for consumers of the profile. This repository does not
provide a general-purpose untrusted-input decoder. Current interoperability tests
use C-Blosc and Zarr readers. Applications SHOULD use maintained, bounded codec
decoders. The Blosc wrapper provides neither authentication nor its own payload
checksum; an outer shard-index checksum does not authenticate chunk contents.

## 9. References

- [BCP 14 / RFC 2119][rfc2119] and
  [RFC 8174][rfc8174]: requirement terminology.
- [C-Blosc 1.21.6 chunk format description][c-blosc-format]:
  the upstream baseline for this profile.
- [C-Blosc 1.21.6 implementation][c-blosc-reference]:
  reference behavior at the same pinned revision.
- [Blosc/Blosc2 chunk description][blosc2-format]:
  additional upstream context; not a moving conformance target.
- [LZ4 block format, v1.10.0][lz4-block-format]:
  compressed LZ4 stream syntax (format document revision 2022-07-31).
- [RFC 8878][rfc8878]: Zstandard frame syntax.

## Appendix A. Verbatim Test Vector

An eight-octet input `00 01 02 03 04 05 06 07`, LZ4, `typesize = 2`, no filter,
and level 0 is encoded by the GPU profile as:

```text
02 01 32 02  08 00 00 00  08 00 00 00  18 00 00 00
00 01 02 03  04 05 06 07
```

The requested API block size can be 16384; its stored value is 8. `cbytes` is
24, and the body has neither an offset table nor length prefix.

## Appendix B. Conformance Evidence

[`test_compress_blosc_gpu.c`][test-compress-blosc-gpu-c] checks header
fields, block offsets, mixed raw/compressed blocks, both filters, short tails,
odd block sizes, whole-chunk fallback, and C-Blosc round trips.
[`test_zarr_readback_gpu.c`][test-zarr-readback-gpu-c] exercises the Zarr
binding. [`test_compress_cpu.c`][test-compress-cpu-c] covers CPU Blosc.
The GPU writer is in [`blosc.frame.cu`][blosc-frame-cu]; the CPU writer
is in [`compress_blosc.c`][compress-blosc-c].

Performance and memory choices are described separately in the
[Blosc performance guide][blosc-performance].

[c-blosc-format]: https://github.com/Blosc/c-blosc/blob/616f4b7343a8479f7e71dd3d7025bd92c9a6bbd0/README_CHUNK_FORMAT.rst
[c-blosc-reference]: https://github.com/Blosc/c-blosc/blob/616f4b7343a8479f7e71dd3d7025bd92c9a6bbd0/blosc/blosc.c
[rfc2119]: https://www.rfc-editor.org/rfc/rfc2119
[rfc8174]: https://www.rfc-editor.org/rfc/rfc8174
[blosc2-format]: https://blosc.org/c-blosc2/format/chunk_format.html
[lz4-block-format]: https://github.com/lz4/lz4/blob/v1.10.0/doc/lz4_Block_format.md
[rfc8878]: https://www.rfc-editor.org/rfc/rfc8878
[test-compress-blosc-gpu-c]: ../tests/test_compress_blosc_gpu.c
[test-zarr-readback-gpu-c]: ../tests/test_zarr_readback_gpu.c
[test-compress-cpu-c]: ../tests/test_compress_cpu.c
[blosc-frame-cu]: ../src/gpu/blosc.frame.cu
[compress-blosc-c]: ../src/cpu/compress_blosc.c
[blosc-performance]: blosc-performance.md
