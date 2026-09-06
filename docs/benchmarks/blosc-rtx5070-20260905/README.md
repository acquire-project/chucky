# RTX 5070 Laptop Blosc measurements

These are retained measurements, not a benchmark to run during a documentation
build. See the [performance guide][blosc-performance] for conclusions,
methodology, measurement limits, and the distinction between calculated
allocation sizes and observed device-memory use.

| Artifact | Contents |
|---|---|
| [summary.csv][summary-csv] | 200 configurations; medians, throughput ranges, compression fold, device usage, and allocation estimates |
| [provenance.json][provenance-json] | Hardware/build settings, source and binary hashes, and run counts |
| [pareto.svg][pareto-plot] | Per-codec throughput/ratio frontiers across all shuffles; random panels zoomed |
| [pareto-all-candidates.svg][pareto-all-candidates] | Same analysis with all candidates visible |
| [pareto-frontier.csv][pareto-frontier] | Per-codec frontier points, with cross-codec frontier membership |
| [pareto-memory-frontier.csv][pareto-memory-frontier] | Cross-codec frontier maximizing speed/ratio and minimizing measured device usage |
| [memory.svg][memory-plot] | Stream device usage and allocation estimate versus block size |
| [memory-estimates.md][memory-estimates] | Memory-versus-block-size tables from the stream estimator results, with bitshuffle |
| [memory-estimates.csv][memory-estimates-csv] | Exact estimated device bytes for all 96 Blosc geometry/codec/filter configurations |
| [pareto.html][pareto-html] | Self-contained local view with point tooltips and tables |
| [plot.py][plot-py] | Standard-library-only plot and frontier generator |

Regenerate the derived artifacts from the repository root:

```sh
python3 docs/benchmarks/blosc-rtx5070-20260905/plot.py
```

The historical and RTX 5080 figures use the same [visual theme](../blosc-figure-style.json)
for codec colors, typography, grids, and graphical legends. Generated SVGs and
HTML embed their styling and remain self-contained.

`summary.csv` is the preserved aggregate from the completed 800-execution sweep;
its old `pareto` column is **per shuffle**, not the frontier shown in the guide.
The generator recomputes membership from throughput and fold. It validates
configuration identities and dominance, and keeps input/chunk groups separate.
The archived CSV uses LF instead of the original CRLF; both hashes are recorded.
Full raw logs and benchmark binaries are not bundled here, and their original
temporary paths in provenance are historical identifiers, not dependencies.

The historical sweep used source `c519a05` and nine validated builds differing
only in the internal block-size constant. It ran from 00:41 to 01:10 UTC on
2026-09-05, before the explicit block-size API. Plot generation does not rebuild
or rerun that experiment.

The memory tables use the sweep's recorded estimator output, not its measured
device deltas. Estimates are independent of input contents; the generator
verifies that both fills agree before combining them. No additional compression
run or standalone memory benchmark is needed. The GPU allocation regression
test checks current estimator accuracy against codec-owned buffers.

Save new measurements under a new date/machine identity rather than replacing
this historical result.

[blosc-performance]: ../../blosc-performance.md
[summary-csv]: summary.csv
[provenance-json]: provenance.json
[pareto-plot]: pareto.svg
[pareto-all-candidates]: pareto-all-candidates.svg
[pareto-frontier]: pareto-frontier.csv
[pareto-memory-frontier]: pareto-memory-frontier.csv
[memory-plot]: memory.svg
[memory-estimates]: memory-estimates.md
[memory-estimates-csv]: memory-estimates.csv
[pareto-html]: pareto.html
[plot-py]: plot.py
