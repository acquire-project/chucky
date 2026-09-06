# L40 and RTX 5070 Laptop frontier comparison

Each row searches all tested block sizes and all three shuffles within one Blosc codec.
Block sizes are in KiB. Membership uses exact medians; min–max ranges in `comparison.csv`
describe observed spread, not confidence intervals. Raw controls are excluded.

The runs differ in source revision, compiler, CUDA toolkit, driver, and GPU.
This compares measured setups and does not isolate a hardware effect.

| Input | Chunk KiB | Codec | RTX 5070 Laptop frontier | L40 frontier | Changed |
|---|---:|---|---|---|---|
| rand | 256 | LZ4 | 4 bit, 8 bit, 16 bit, 32 bit, 256 bit | 4 bit, 8 bit, 128 bit, 256 bit | yes |
| rand | 256 | Zstd | 16 bit, 32 bit, 64 bit, 128 bit | 256 bit | yes |
| rand | 1024 | LZ4 | 4 bit, 8 bit, 16 bit, 32 bit, 64 bit, 1024 bit | 4 bit, 8 bit, 256 bit | yes |
| rand | 1024 | Zstd | 16 bit, 32 bit, 64 bit, 128 bit, 1024 bit | 1024 bit | yes |
| xor | 256 | LZ4 | 256 bit | 16 byte, 32 byte, 64 byte, 128 byte, 256 bit | yes |
| xor | 256 | Zstd | 64 bit, 128 bit, 256 bit | 32 bit, 128 bit, 256 bit | yes |
| xor | 1024 | LZ4 | 1024 bit | 4 bit, 4 byte, 8 byte, 32 byte, 64 byte, 128 byte, 256 bit, 512 bit, 1024 bit | yes |
| xor | 1024 | Zstd | 64 bit, 128 bit, 256 bit, 512 bit, 1024 bit | 32 bit, 64 bit, 128 bit, 256 bit, 512 bit, 1024 bit | yes |
