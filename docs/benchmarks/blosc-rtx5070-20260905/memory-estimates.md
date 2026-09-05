# GPU memory allocation estimates

Generated from the estimator results in [summary.csv][summary-csv].
These are calculated allocation sizes, not device-memory measurements.
The estimates agree across both input fills. No compression rerun is needed.

Build: nvCOMP 5.3.0.16; source and configuration: [provenance.json][provenance-json].
Bitshuffle throughout; whole-stream device memory, excluding runtime overhead.

## 256 KiB chunks, 144 MiB padded batch

| Block KiB | LZ4 device GiB | Zstd device GiB |
|---:|---:|---:|
| 4 | 2.502 | 1.723 |
| 8 | 1.939 | 1.730 |
| 16 | 1.657 | 1.746 |
| 32 | 1.516 | 1.780 |
| 64 | 1.446 | 1.849 |
| 128 | 1.411 | 1.849 |
| 256 | 1.393 | 1.849 |

## 1024 KiB chunks, 288 MiB padded batch

| Block KiB | LZ4 device GiB | Zstd device GiB |
|---:|---:|---:|
| 4 | 4.754 | 3.082 |
| 8 | 3.628 | 3.085 |
| 16 | 3.064 | 3.100 |
| 32 | 2.783 | 3.134 |
| 64 | 2.642 | 3.202 |
| 128 | 2.572 | 3.202 |
| 256 | 2.536 | 3.202 |
| 512 | 2.519 | 3.202 |
| 1024 | 2.510 | 3.202 |

[All filters and exact byte totals][memory-estimates-csv].

See the [performance guide][blosc-performance] for sizing terms,
runtime headroom, and the distinction between estimates and measurements.

[summary-csv]: summary.csv
[provenance-json]: provenance.json
[memory-estimates-csv]: memory-estimates.csv
[blosc-performance]: ../../blosc-performance.md
