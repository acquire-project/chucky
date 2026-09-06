#!/usr/bin/env python3
"""Plot measured GPU Blosc throughput/compression Pareto frontiers."""

import csv
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
STYLE = json.loads((ROOT.parent / "blosc-figure-style.json").read_text(encoding="utf-8"))
COLORS = STYLE["colors"]
PANELS = [("xor", 256), ("xor", 1024), ("rand", 256), ("rand", 1024)]
WIDTH, HEIGHT = 1500, 1220


def dominates(a, b):
    return (a["speed"] >= b["speed"] and a["fold"] >= b["fold"]
            and (a["speed"] > b["speed"] or a["fold"] > b["fold"]))


def frontier(rows):
    return sorted((r for r in rows if not any(dominates(s, r) for s in rows)),
                  key=lambda r: (r["fold"], -r["speed"]))


def text(x, y, value, cls="", anchor="start", color=None):
    style = f' style="fill:{color}"' if color else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" '
            f'text-anchor="{anchor}"{style}>{html.escape(str(value))}</text>')


def line(x1, y1, x2, y2, color, width=1, extra=""):
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>')


def marker(x, y, shape, color, radius=4, opacity=1, title=""):
    attrs = (f'fill="{color}" stroke="{color}" opacity="{opacity}" '
             'stroke-width="1.4"')
    tip = f'<title>{html.escape(title)}</title>'
    if shape == "bit":
        points = f'{x},{y-radius} {x-radius},{y+radius} {x+radius},{y+radius}'
        return f'<polygon points="{points}" {attrs}>{tip}</polygon>'
    if shape == "byte":
        return f'<circle cx="{x}" cy="{y}" r="{radius}" {attrs}>{tip}</circle>'
    if shape == "raw":
        points = f'{x},{y-radius} {x-radius},{y} {x},{y+radius} {x+radius},{y}'
        return (f'<polygon points="{points}" fill="white" stroke="{color}" '
                f'stroke-width="1.7">{tip}</polygon>')
    return (f'<rect x="{x-radius}" y="{y-radius}" width="{radius*2}" '
            f'height="{radius*2}" {attrs}>{tip}</rect>')


def legend(memory=False):
    """Use actual line and marker samples for every visual encoding."""
    neutral = STYLE["muted"]
    out = ['<g role="group" aria-label="Figure legend">',
           f'<rect x="48" y="96" width="1404" height="{60 if memory else 96}" rx="8" '
           f'fill="{STYLE["legend_fill"]}" stroke="{STYLE["legend_border"]}"/>']
    for x0, codec in [(72, "blosc-lz4"), (212, "blosc-zstd")]:
        out += [line(x0, 126, x0 + 32, 126, COLORS[codec], 3),
                text(x0 + 43, 131, codec.removeprefix("blosc-").upper(), "legend-label")]
    if memory:
        out += [line(390, 126, 430, 126, neutral, 2.5),
                marker(410, 126, "byte", neutral, 4),
                text(443, 131, "Observed device delta", "legend-label"),
                line(730, 126, 770, 126, neutral, 2.5, 'stroke-dasharray="7 5"'),
                text(783, 131, "Explicit allocation estimate", "legend-label")]
    else:
        for x0, shape, label in [(390, "none", "No shuffle"), (570, "byte", "Byte shuffle"),
                                 (760, "bit", "Bitshuffle"), (940, "raw", "Raw codec control")]:
            out += [marker(x0, 126, shape, neutral, 5), text(x0 + 17, 131, label, "legend-label")]
        out += [line(72, 166, 112, 166, neutral, 2.5),
                marker(92, 166, "bit", neutral, 5), text(125, 171, "Per-codec frontier", "legend-label"),
                marker(390, 166, "bit", neutral, 5, .23), text(407, 171, "Dominated candidate", "legend-label"),
                line(760, 155, 760, 177, neutral, 1.4),
                line(755, 155, 765, 155, neutral, 1.4), line(755, 177, 765, 177, neutral, 1.4),
                text(777, 171, "Min–max of 3 runs", "legend-label")]
    return "\n".join(out + ['</g>'])


def tooltip(row):
    block = f'{row["block_kib"]} KiB / {row["shuffle"]}' if row["block_kib"] else "raw control"
    return (f'{row["codec"]} / {block}: {row["speed"]:.5f} GiB/s '
            f'(min–max {row["lo"]:.5f}–{row["hi"]:.5f}); '
            f'{row["fold"]:.5f}×; device allocation delta {row["device_gib"]:.3f} GiB')


def separated_labels(points, top, bottom):
    ordered = sorted(points, key=lambda item: item[1])
    ys = []
    for _, cy in ordered:
        ys.append(max(cy, ys[-1] + 22 if ys else top))
    if ys and ys[-1] > bottom:
        ys[-1] = bottom
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - 22)
    return [(row, cy) for (row, _), cy in zip(ordered, ys)]


def panel(rows, fill, chunk, index, zoom):
    ox, oy = 30 + (index % 2) * 745, 250 + (index // 2) * 430
    left, right, top, bottom = ox + 62, ox + 677, oy + 48, oy + 353
    group = [r for r in rows if r["fill"] == fill and r["chunk_kib"] == chunk]
    fronts = {codec: frontier([r for r in group if r["codec"] == codec]) for codec in COLORS}
    focused = zoom and fill == "rand"
    if fill == "xor":
        xmin, xmax, ticks = 1, 500, [1, 2, 5, 10, 20, 50, 100, 200, 500]
        transform = math.log10
    elif focused:
        if chunk == 256:
            xmin, xmax, ticks = 1.367, 1.39, [1.37, 1.375, 1.38, 1.385, 1.39]
        else:
            xmin, xmax, ticks = 1.47, 1.497, [1.47, 1.475, 1.48, 1.485, 1.49, 1.495]
        transform = float
    else:
        xmin, xmax, ticks = 1, 1.55, [1, 1.1, 1.2, 1.3, 1.4, 1.5]
        transform = float

    def x(value):
        return left + (transform(value) - transform(xmin)) / (transform(xmax) - transform(xmin)) * (right - left)

    def y(value):
        return bottom - value / 6 * (bottom - top)

    title = "Repetitive XOR" if fill == "xor" else "12-bit random"
    size = "256 KiB" if chunk == 256 else "1 MiB"
    scale = "log ratio axis" if fill == "xor" else "zoomed ratio axis" if focused else "linear ratio axis"
    out = [text(ox + 18, oy + 7, f'{title} · {size} Zarr chunks', "panel-title"),
           text(ox + 18, oy + 28, scale, "muted")]
    for tick in range(7):
        out += [line(left, y(tick), right, y(tick), "#e3e8ec"),
                text(left - 12, y(tick) + 4, tick, anchor="end")]
    for tick in ticks:
        out += [line(x(tick), top, x(tick), bottom, STYLE["grid"]),
                text(x(tick), bottom + 24, f'{tick:g}×', anchor="middle")]
    out += [line(left, bottom, right, bottom, "#a3adb7"),
            text((left + right) / 2, bottom + 52, "Compression fold →", "axis", "middle"),
            f'<text transform="translate({ox+13},{(top+bottom)/2}) rotate(-90)" '
            'text-anchor="middle" class="axis">Input throughput (GiB/s) →</text>']
    visible = lambda r: xmin <= r["fold"] <= xmax
    front_ids = {r["id"] for front in fronts.values() for r in front}
    for row in group:
        if visible(row) and row["codec"] in COLORS and row["id"] not in front_ids:
            out.append(marker(x(row["fold"]), y(row["speed"]), row["shuffle"],
                              COLORS[row["codec"]], opacity=.23, title=tooltip(row)))
        elif visible(row) and not row["block_kib"]:
            color = COLORS[f'blosc-{row["codec"]}']
            out.append(marker(x(row["fold"]), y(row["speed"]), "raw", color, 5, title=tooltip(row)))
    for codec, front in fronts.items():
        color = COLORS[codec]
        points = " ".join(f'{x(r["fold"]):.2f},{y(r["speed"]):.2f}' for r in front)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in front:
            cx = x(row["fold"])
            out += [line(cx, y(row["lo"]), cx, y(row["hi"]), color, 1.4),
                    line(cx - 4, y(row["lo"]), cx + 4, y(row["lo"]), color, 1.4),
                    line(cx - 4, y(row["hi"]), cx + 4, y(row["hi"]), color, 1.4),
                    marker(cx, y(row["speed"]), row["shuffle"], color, 6, title=tooltip(row))]
        label_offset = 26 if fill == "xor" and codec == "blosc-lz4" else 14
        labels = separated_labels([(r, y(r["speed"]) - label_offset) for r in front], top + 10, bottom - 10)
        for row, ly in labels:
            cx, cy = x(row["fold"]), y(row["speed"])
            lx, anchor = cx - 12, "end"
            if cx < left + 85:
                lx, anchor = cx + 12, "start"
            out += [line(cx, cy, lx + (3 if anchor == "end" else -3), ly - 4, color, .7, 'opacity=".65"'),
                    text(lx, ly, f'{row["block_kib"]} KiB', "point-label", anchor, color)]
    return "\n".join(out)


def figure(rows, zoom):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
           f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
           '<title id="title">GPU Blosc throughput and compression Pareto frontiers</title>',
           '<desc id="desc">Four panels compare LZ4 and Zstd for two input patterns and two Zarr chunk sizes. '
           'Both axes are maximized. Frontiers are calculated separately per codec across block sizes and shuffle modes.</desc>',
           f'<style>{STYLE["css"]}</style>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
           text(48, 43, "GPU Blosc: throughput / compression Pareto frontiers", "title"),
           text(48, 72, "RTX 5070 Laptop GPU · nvCOMP 5.3 · driver 595.99.02 · sweep 2026-09-05 UTC", "subtitle"),
           legend(),
           text(48, 217, "Labels: block KiB · per-codec frontier across all filters · higher and farther right is better", "muted")]
    for index, (fill, chunk) in enumerate(PANELS):
        out.append(panel(rows, fill, chunk, index, zoom))
    out += [text(48, 1140, "Frontier = no other tested Blosc setting of the same codec improves either median objective without worsening the other.", "note"),
            text(48, 1162, "Fold = padded chunk bytes / encoded bytes; compare within each panel. Discard sink + synthetic inputs, not a storage benchmark.", "note"),
            text(48, 1184, "GPU memory is not an objective here (see hover values / CSV). Lines guide the eye; intermediate settings were not measured.", "note"),
            text(48, 1206, "Random panels zoom to the frontiers; off-scale candidates / raw controls are omitted. Timing ranges are not confidence intervals."
                 if zoom else "All measured candidates shown. Exact-median frontier membership can be sensitive to small timing or compression differences.", "note"),
            '</svg>']
    return "\n".join(out)


def main():
    rows = []
    with (ROOT / "summary.csv").open(newline="") as stream:
        for summary in csv.DictReader(stream):
            assert int(summary["repeats"]) == 3
            key = "__".join(summary[k] for k in ("fill", "chunk_kib", "block_kib", "codec", "shuffle"))
            rows.append(dict(id=key, fill=summary["fill"], chunk_kib=int(summary["chunk_kib"]),
                             block_kib=int(summary["block_kib"]), codec=summary["codec"], shuffle=summary["shuffle"],
                             speed=float(summary["throughput_median_gibs"]), lo=float(summary["throughput_min_gibs"]),
                             hi=float(summary["throughput_max_gibs"]), fold=float(summary["compression_fold"]),
                             device_gib=float(summary["device_gib"]), estimate_gib=float(summary["estimated_total_gib"])))
    assert all(0 < r["lo"] <= r["speed"] <= r["hi"] <= 6 and r["fold"] > 0 for r in rows)
    assert len(rows) == 200 and len({r["id"] for r in rows}) == 200
    exported, tables = [], []
    for fill, chunk in PANELS:
        group = [r for r in rows if r["fill"] == fill and r["chunk_kib"] == chunk and r["block_kib"]]
        overall = {r["id"] for r in frontier(group)}
        tables.append(f'<h2>{fill.upper()} · {chunk} KiB chunks</h2><table><thead><tr>'
                      '<th>Codec</th><th>Block KiB</th><th>Shuffle</th><th>GiB/s</th>'
                      '<th>Min–max</th><th>Fold</th><th>Device GiB</th><th>Across both codecs?</th>'
                      '</tr></thead><tbody>')
        for codec in COLORS:
            candidates = [r for r in group if r["codec"] == codec]
            front = frontier(candidates)
            assert all(r["shuffle"] == "bit" for r in front)
            assert all(r in front or any(dominates(s, r) for s in front) for r in candidates)
            for row in front:
                exported.append(dict(row, overall_frontier=row["id"] in overall))
                tables.append(f'<tr><td>{codec}</td><td>{row["block_kib"]}</td><td>{row["shuffle"]}</td>'
                              f'<td>{row["speed"]:.5f}</td><td>{row["lo"]:.5f}–{row["hi"]:.5f}</td>'
                              f'<td>{row["fold"]:.5f}</td><td>{row["device_gib"]:.3f}</td>'
                              f'<td>{"yes" if row["id"] in overall else "no"}</td></tr>')
        tables.append('</tbody></table>')
    assert len(frontier([dict(speed=1, fold=2), dict(speed=1, fold=2)])) == 2
    assert frontier([dict(speed=1, fold=2), dict(speed=1, fold=3)]) == [dict(speed=1, fold=3)]
    assert len(frontier([dict(speed=1, fold=3), dict(speed=2, fold=2)])) == 2
    assert len(frontier([dict(speed=1, fold=2), dict(speed=2, fold=3)])) == 1
    with (ROOT / "pareto-frontier.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(exported)
    for name, zoom in [("pareto", True), ("pareto-all-candidates", False)]:
        svg = figure(rows, zoom)
        ET.fromstring(svg)
        (ROOT / f"{name}.svg").write_text(svg, encoding="utf-8", newline="\n")
    svg = memory_figure(rows)
    ET.fromstring(svg)
    (ROOT / "memory.svg").write_text(svg, encoding="utf-8", newline="\n")
    write_memory_estimates(rows)
    with (ROOT / "pareto-memory-frontier.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for fill, chunk in PANELS:
            candidates = [r for r in rows if r["fill"] == fill and r["chunk_kib"] == chunk and r["block_kib"]]
            front = [r for r in candidates if not any(dominates_memory(s, r) for s in candidates)]
            assert all(r in front or any(dominates_memory(s, r) for s in front) for r in candidates)
            writer.writerows(front)
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>GPU Blosc Pareto frontiers</title><style>'
                'body{margin:0;background:white;font:15px system-ui,sans-serif;color:#283747}'
                'svg{display:block;width:100%;height:auto}main{max-width:1500px;margin:auto}'
                'section{padding:0 48px 40px}a{color:#1477ba}table{border-collapse:collapse;width:100%;margin:20px 0}'
                'th,td{text-align:right;padding:8px;border-bottom:1px solid #e3e8ec}'
                'th:first-child,td:first-child{text-align:left}h2{margin-top:32px}'
                '</style><main>' + figure(rows, True) + '<section>'
                '<p><a href="pareto.svg">Frontier SVG</a> · <a href="pareto-all-candidates.svg">All candidates SVG</a> · '
                '<a href="pareto-frontier.csv">Frontier CSV</a> · <a href="../../blosc-performance.md">Sweep methodology</a></p>'
                '<p>Hover over points for measurements and device-memory allocation deltas. '
                'Frontiers are recomputed across all shuffle modes per codec, not copied from the per-shuffle CSV flags. '
                'Raw codec controls are references only and do not participate in Blosc frontiers. '
                'The last column also checks dominance across both Blosc codecs. '
                'Only measured medians determine membership; no statistical significance or interpolation is implied.</p>'
                + "\n".join(tables) + '</section></main></html>')
    (ROOT / "pareto.html").write_text(document, encoding="utf-8", newline="\n")
    print(f'Validated {len(rows)} configurations; exported {len(exported)} per-codec frontier settings.')
    print(ROOT / "pareto.html")


def dominates_memory(a, b):
    return (a["speed"] >= b["speed"] and a["fold"] >= b["fold"] and a["device_gib"] <= b["device_gib"]
            and (a["speed"] > b["speed"] or a["fold"] > b["fold"] or a["device_gib"] < b["device_gib"]))


def write_memory_estimates(rows):
    estimates = {}
    for row in rows:
        if not row["block_kib"]:
            continue
        key = tuple(row[k] for k in ("chunk_kib", "codec", "shuffle", "block_kib"))
        byte_value = row["estimate_gib"] * 1024**3
        assert math.isfinite(byte_value) and byte_value > 0 and byte_value.is_integer()
        value = int(byte_value)
        assert key not in estimates or estimates[key] == value
        estimates[key] = value
    assert len(estimates) == 96
    with (ROOT / "memory-estimates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("chunk_kib", "codec", "shuffle", "block_kib", "batch_chunks", "estimated_device_bytes"))
        for key, value in sorted(estimates.items()):
            writer.writerow((*key, 576 if key[0] == 256 else 288, value))
    lines = ["# GPU memory allocation estimates", "",
             "Generated from the estimator results in [summary.csv][summary-csv].",
             "These are calculated allocation sizes, not device-memory measurements.",
             "The estimates agree across both input fills. No compression rerun is needed.", "",
             "Build: nvCOMP 5.3.0.16; source and configuration: [provenance.json][provenance-json].",
             "Bitshuffle throughout; whole-stream device memory, excluding runtime overhead.", ""]
    for chunk in (256, 1024):
        lines += [f"## {chunk} KiB chunks, {144 if chunk == 256 else 288} MiB padded batch", "",
                  "| Block KiB | LZ4 device GiB | Zstd device GiB |", "|---:|---:|---:|"]
        blocks = sorted({key[3] for key in estimates if key[0] == chunk})
        for block in blocks:
            lz4 = estimates[chunk, "blosc-lz4", "bit", block] / 1024**3
            zstd = estimates[chunk, "blosc-zstd", "bit", block] / 1024**3
            lines.append(f"| {block} | {lz4:.3f} | {zstd:.3f} |")
        lines.append("")
    lines += ["[All filters and exact byte totals][memory-estimates-csv].", "",
              "See the [performance guide][blosc-performance] for sizing terms,",
              "runtime headroom, and the distinction between estimates and measurements.", "",
              "[summary-csv]: summary.csv",
              "[provenance-json]: provenance.json",
              "[memory-estimates-csv]: memory-estimates.csv",
              "[blosc-performance]: ../../blosc-performance.md", ""]
    (ROOT / "memory-estimates.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def memory_figure(rows):
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="620" viewBox="0 0 1500 620" role="img">',
           '<title>Blosc block size and stream GPU memory</title>',
           f'<style>{STYLE["css"]}</style>',
           '<rect width="1500" height="620" fill="white"/>',
           text(48, 43, "Block size and GPU memory", "title"),
           text(48, 72, "RTX 5070 Laptop GPU · nvCOMP 5.3 · XOR input · bitshuffle · whole-stream device memory", "subtitle"),
           legend(memory=True)]
    for p, chunk in enumerate((256, 1024)):
        left, right, top, bottom = 92 + 745*p, 707 + 745*p, 224, 505
        out.append(text(left - 44, 198, f'{chunk} KiB chunks · {144 if chunk == 256 else 288} MiB padded batch', "panel-title"))
        def x(block):
            return left + (math.log2(block) - 2) / (math.log2(chunk) - 2) * (right - left)
        def y(value):
            return bottom - value / 5 * (bottom - top)
        for tick in range(6):
            out += [line(left, y(tick), right, y(tick), "#e3e8ec"), text(left - 12, y(tick) + 4, tick, anchor="end")]
        for block in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            if block <= chunk:
                out.append(text(x(block), bottom + 25, block, anchor="middle"))
        out += [text((left + right)/2, bottom + 52, "Blosc block size (KiB, log₂)", anchor="middle"),
                f'<text transform="translate({left-53},{(top+bottom)/2}) rotate(-90)" text-anchor="middle">Device GiB</text>']
        for codec, color in COLORS.items():
            group = sorted((r for r in rows if r["fill"] == "xor" and r["chunk_kib"] == chunk
                            and r["shuffle"] == "bit" and r["codec"] == codec), key=lambda r: r["block_kib"])
            for field, dash in (("device_gib", ""), ("estimate_gib", 'stroke-dasharray="7 5"')):
                points = " ".join(f'{x(r["block_kib"]):.2f},{y(r[field]):.2f}' for r in group)
                out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>')
            for row in group:
                out.append(marker(x(row["block_kib"]), y(row["device_gib"]), "byte", color, title=tooltip(row)))
            last = group[-1]
            out.append(text(right - 8, y(last["device_gib"]) - 12, codec.removeprefix("blosc-").upper(), anchor="end", color=color))
    out += [text(48, 600, "Device delta is measured before creation versus after the run with the stream alive; it is not a sampled peak. Runtime overhead is not estimated.", "note"), '</svg>']
    return "\n".join(out)


if __name__ == "__main__":
    main()
