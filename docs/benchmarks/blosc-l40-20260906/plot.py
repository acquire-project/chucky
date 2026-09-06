#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.11.1"]
# ///

import argparse
import base64
import csv
import gzip
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / "blosc-rtx5070-20260905"
GIB = 1024**3
GROUPS = [(fill, chunk) for fill in ("rand", "xor") for chunk in (256, 1024)]
PANELS = [(fill, chunk) for fill in ("xor", "rand") for chunk in (256, 1024)]
COLORS = {"blosc-lz4": "#1477ba", "blosc-zstd": "#c34d28"}
TEXT_COLOR = "#283747"
MUTED_COLOR = "#637281"
CODEC_NAMES = {"blosc-lz4": "LZ4", "blosc-zstd": "Zstd"}
MARKERS = {"none": "s", "byte": "o", "bit": "^"}
KEYS = ("fill", "chunk_kib", "block_kib", "codec", "shuffle")


def key(row):
    return tuple(str(row[field]) for field in KEYS)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_summary(path):
    rows = []
    for source in read_csv(path):
        row = {field: source[field] for field in ("fill", "codec", "shuffle")}
        row.update(
            {
                field: int(source[field])
                for field in ("chunk_kib", "block_kib", "repeats")
            }
        )
        row.update(
            speed=float(source["throughput_median_gibs"]),
            minimum=float(source["throughput_min_gibs"]),
            maximum=float(source["throughput_max_gibs"]),
            span=float(source["throughput_span_pct"]),
            fold=float(source["compression_fold"]),
            memory=float(source["device_gib"]),
            estimate=float(
                source.get("estimated_device_gib", source.get("estimated_total_gib"))
            ),
        )
        if "estimated_device_bytes" in source:
            row["estimate_bytes"] = int(source["estimated_device_bytes"])
            row["overhead_bytes"] = float(source["device_overhead_bytes"])
        rows.append(row)
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate configurations: {path}")
    return rows


def dominates(a, b, memory=False):
    left = (a["speed"], a["fold"], -a["memory"]) if memory else (a["speed"], a["fold"])
    right = (b["speed"], b["fold"], -b["memory"]) if memory else (b["speed"], b["fold"])
    return all(x >= y for x, y in zip(left, right)) and any(
        x > y for x, y in zip(left, right)
    )


def frontier(rows, memory=False):
    return sorted(
        (
            row
            for row in rows
            if not any(dominates(other, row, memory) for other in rows)
        ),
        key=lambda row: (row["fold"], -row["speed"], row["block_kib"], row["shuffle"]),
    )


def group(rows, fill, chunk, codec=None):
    return [
        row
        for row in rows
        if row["fill"] == fill
        and row["chunk_kib"] == chunk
        and (codec is None or row["codec"] == codec)
    ]


def mark_frontiers(rows):
    for row in rows:
        row.update(
            per_codec_frontier=False, cross_codec_frontier=False, memory_frontier=False
        )
    for fill, chunk in GROUPS:
        candidates = [row for row in group(rows, fill, chunk) if row["codec"] in COLORS]
        for codec in COLORS:
            for row in frontier([r for r in candidates if r["codec"] == codec]):
                row["per_codec_frontier"] = True
        for row in frontier(candidates):
            row["cross_codec_frontier"] = True
        for row in frontier(candidates, memory=True):
            row["memory_frontier"] = True


def validate(rows, previous):
    provenance = json.loads((ROOT / "provenance.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    build = json.loads((ROOT / "build.json").read_text())
    validation = json.loads((ROOT / "validation.json").read_text())
    if provenance["build"] != build or provenance["manifest"] != manifest:
        raise ValueError("Provenance does not match retained build inputs")
    for field, name in (
        ("manifest_sha256", "manifest.json"),
        ("harness_sha256", "sweep.py"),
        ("patch_sha256", "benchmark-controls.patch"),
    ):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != build[field]:
            raise ValueError(f"Collection input hash differs: {name}")
    if (
        hashlib.sha256((HISTORY / "summary.csv").read_bytes()).hexdigest()
        != validation["historical_summary_sha256"]
    ):
        raise ValueError("The historical summary changed")
    expected = {key(row) for row in manifest["configurations"]}
    if (
        not provenance["complete"]
        or expected != {key(row) for row in rows}
        or len(expected) != 200
    ):
        raise ValueError("The full 200-configuration sweep is incomplete")
    if expected != {key(row) for row in previous}:
        raise ValueError("Historical configurations do not match")
    samples = read_csv(ROOT / "runs.csv")
    if len(samples) != len(rows) * (manifest["warmups"] + manifest["repeats"]):
        raise ValueError("Execution count differs from the manifest")
    with gzip.open(ROOT / "results.jsonl.gz", "rb") as stream:
        raw_bytes = stream.read()
    if hashlib.sha256(raw_bytes).hexdigest() != validation["results_jsonl_sha256"]:
        raise ValueError("Raw measurement hash differs")
    records = [json.loads(line) for line in raw_bytes.splitlines()]
    if len(records) != len(samples) or provenance["completed"] != len(records):
        raise ValueError("Raw records and compact measurements disagree")
    for record, sample in zip(records, samples):
        if key(record["config"]) != key(sample) or record["repeat"] != int(
            sample["repeat"]
        ):
            raise ValueError("Execution identities differ")
        if (
            record["warmup"] != (record["repeat"] == 0)
            or str(record["warmup"]) != sample["warmup"]
        ):
            raise ValueError("Warmup flag differs from repetition identity")
        expected_shape = (
            [8, 1, 128, 128]
            if record["config"]["chunk_kib"] == 256
            else [16, 1, 128, 256]
        )
        expected_chunks = 576 if record["config"]["chunk_kib"] == 256 else 288
        actual, result = record["actual"], record["result"]
        if (
            actual["chunks_per_epoch"] != expected_chunks
            or actual["padded_batch_bytes"]
            != expected_chunks * record["config"]["chunk_kib"] * 1024
        ):
            raise ValueError("Changed epoch or padded batch size")
        if (
            result["chunks_per_epoch"] != expected_chunks
            or result["total_chunks"]
            != math.ceil(100 / expected_shape[0]) * expected_chunks
        ):
            raise ValueError("Changed measured chunk count")
        if result["worker_threads"] != 3 or not math.isclose(
            result["input_gib"], 1.7578125, rel_tol=1e-5
        ):
            raise ValueError("Changed worker count or input size")
        if any(
            result[field] != record["config"][field] for field in ("shuffle", "level")
        ):
            raise ValueError("Returned codec settings differ from the request")
        if (
            record["config"]["block_kib"]
            and result["blosc_block_bytes"] != record["config"]["block_kib"] * 1024
        ):
            raise ValueError("Returned Blosc block size differs from the request")
        if (
            record["result"]["status"] != "pass"
            or record["actual"]["chunk_shape"] != expected_shape
        ):
            raise ValueError("Failed execution or changed chunk shape")
        if record["actual"]["epochs_per_batch"] != 1:
            raise ValueError("Changed batch geometry")
        for field in (
            "throughput_in_gibs",
            "compression_fold",
            "memory_device_used_bytes",
            "memory_estimate_total_bytes",
            "memory_device_overhead_bytes",
        ):
            if float(sample[field]) != record["result"][field]:
                raise ValueError(
                    f"Compact measurement differs from raw output: {field}"
                )
    for row in rows:
        all_samples = [sample for sample in samples if key(sample) == key(row)]
        if sorted(int(sample["repeat"]) for sample in all_samples) != list(
            range(manifest["repeats"] + 1)
        ):
            raise ValueError(f"Missing or duplicate repetitions: {key(row)}")
        selected = [sample for sample in all_samples if sample["warmup"] == "False"]
        if len(selected) != row["repeats"] or row["repeats"] != manifest["repeats"]:
            raise ValueError(f"Wrong measurement count: {key(row)}")
        values = [float(sample["throughput_in_gibs"]) for sample in selected]
        checks = {
            "speed": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "span": 100 * (max(values) - min(values)) / statistics.median(values),
            "fold": float(selected[0]["compression_fold"]),
            "memory": statistics.median(
                int(s["memory_device_used_bytes"]) for s in selected
            )
            / GIB,
            "estimate": int(selected[0]["memory_estimate_total_bytes"]) / GIB,
            "overhead_bytes": statistics.median(
                int(s["memory_device_overhead_bytes"]) for s in selected
            ),
        }
        if any(
            not math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-12)
            for field, value in checks.items()
        ):
            raise ValueError(f"Summary does not match measurements: {key(row)}")
        for sample in selected:
            if float(sample["compression_fold"]) != row["fold"]:
                raise ValueError("Compression ratio varied across repetitions")
            if int(sample["memory_estimate_total_bytes"]) != row["estimate_bytes"]:
                raise ValueError("Allocation estimate varied across repetitions")
            if int(sample["memory_device_used_bytes"]) - row["estimate_bytes"] != int(
                sample["memory_device_overhead_bytes"]
            ):
                raise ValueError(
                    "Memory residual differs from observed minus estimated bytes"
                )
    return provenance, samples


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_row(row):
    return dict(
        fill=row["fill"],
        chunk_kib=row["chunk_kib"],
        codec=row["codec"],
        shuffle=row["shuffle"],
        block_kib=row["block_kib"],
        throughput_median_gibs=row["speed"],
        throughput_min_gibs=row["minimum"],
        throughput_max_gibs=row["maximum"],
        compression_fold=row["fold"],
        device_gib=row["memory"],
        per_codec_frontier=row["per_codec_frontier"],
        cross_codec_frontier=row["cross_codec_frontier"],
        memory_frontier=row["memory_frontier"],
    )


def block_list(rows):
    return ", ".join(
        f"{row['block_kib']} {row['shuffle']}"
        for row in sorted(rows, key=lambda r: (r["block_kib"], r["shuffle"]))
    )


def make_tables(rows, previous, samples):
    write_csv(
        "pareto-frontier.csv",
        [csv_row(row) for row in rows if row["per_codec_frontier"]],
    )
    write_csv(
        "pareto-memory-frontier.csv",
        [csv_row(row) for row in rows if row["memory_frontier"]],
    )
    old = {key(row): row for row in previous}
    comparisons = []
    for row in rows:
        before = old[key(row)]
        result = {field: row[field] for field in KEYS}
        for prefix, point in (("l40", row), ("rtx5070", before)):
            result.update(
                {
                    f"{prefix}_{field}": value
                    for field, value in csv_row(point).items()
                    if field not in KEYS
                }
            )
        result["throughput_l40_over_rtx5070"] = row["speed"] / before["speed"]
        result["fold_change_pct"] = 100 * (row["fold"] / before["fold"] - 1)
        comparisons.append(result)
    write_csv("comparison.csv", comparisons)
    memory_rows = []
    for row in rows:
        if row["fill"] != "xor" or row["codec"] not in COLORS:
            continue
        other = next(
            r
            for r in group(rows, "rand", row["chunk_kib"], row["codec"])
            if r["block_kib"] == row["block_kib"] and r["shuffle"] == row["shuffle"]
        )
        if row["estimate_bytes"] != other["estimate_bytes"]:
            raise ValueError("Memory allocation estimates depend on the fill")
        memory_rows.append(
            dict(
                chunk_kib=row["chunk_kib"],
                codec=row["codec"],
                shuffle=row["shuffle"],
                block_kib=row["block_kib"],
                estimated_device_bytes=row["estimate_bytes"],
                estimated_device_gib=row["estimate"],
                xor_observed_device_gib=row["memory"],
                xor_overhead_mib=row["overhead_bytes"] / 1024**2,
            )
        )
    if len(memory_rows) != 96:
        raise ValueError("Expected 96 distinct Blosc memory configurations")
    write_csv("memory-estimates.csv", memory_rows)
    lines = [
        "# L40 stream memory by Blosc block size",
        "",
        "Estimates are explicit device allocations for the complete stream. Observations are",
        "median device-memory deltas on XOR input; they are not sampled peaks. The estimates",
        "agree across random and XOR fills. Pinned host memory is excluded.",
        "",
    ]
    for chunk in (256, 1024):
        for shuffle in MARKERS:
            lines += [
                f"## {chunk} KiB chunks, {shuffle} shuffle",
                "",
                "| Block KiB | LZ4 estimate GiB | LZ4 observed GiB | LZ4 residual MiB | Zstd estimate GiB | Zstd observed GiB | Zstd residual MiB |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
            candidates = [
                r
                for r in memory_rows
                if r["chunk_kib"] == chunk and r["shuffle"] == shuffle
            ]
            for block in sorted({r["block_kib"] for r in candidates}):
                cells = [str(block)]
                for codec in COLORS:
                    row = next(
                        r
                        for r in candidates
                        if r["block_kib"] == block and r["codec"] == codec
                    )
                    cells.extend(
                        [
                            f"{row['estimated_device_gib']:.3f}",
                            f"{row['xor_observed_device_gib']:.3f}",
                            f"{row['xor_overhead_mib']:.1f}",
                        ]
                    )
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    (ROOT / "memory-estimates.md").write_text("\n".join(lines))
    changes = []
    lines = [
        "# L40 and RTX 5070 Laptop frontier comparison",
        "",
        "Each row searches all tested block sizes and all three shuffles within one Blosc codec.",
        "Block sizes are in KiB. Membership uses exact medians; min–max ranges in `comparison.csv`",
        "describe observed spread, not confidence intervals. Raw controls are excluded.",
        "",
        "The runs differ in source revision, compiler, CUDA toolkit, driver, and GPU.",
        "This compares measured setups and does not isolate a hardware effect.",
        "",
        "| Input | Chunk KiB | Codec | RTX 5070 Laptop frontier | L40 frontier | Changed |",
        "|---|---:|---|---|---|---|",
    ]
    for fill, chunk in GROUPS:
        for codec in COLORS:
            before = frontier(group(previous, fill, chunk, codec))
            after = frontier(group(rows, fill, chunk, codec))
            changed = {key(r) for r in before} != {key(r) for r in after}
            changes.append(
                dict(
                    fill=fill,
                    chunk_kib=chunk,
                    codec=codec,
                    changed=changed,
                    rtx5070=block_list(before),
                    l40=block_list(after),
                )
            )
            lines.append(
                f"| {fill} | {chunk} | {CODEC_NAMES[codec]} | {block_list(before)} | {block_list(after)} | {'yes' if changed else 'no'} |"
            )
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    measured = [sample for sample in samples if sample["warmup"] == "False"]
    residuals = []
    for fill, chunk in GROUPS:
        for codec in COLORS:
            selected = [
                s
                for s in measured
                if s["fill"] == fill
                and int(s["chunk_kib"]) == chunk
                and s["codec"] == codec
            ]
            values = [
                int(s["memory_device_overhead_bytes"]) / 1024**2 for s in selected
            ]
            residuals.append(
                dict(
                    fill=fill,
                    chunk_kib=chunk,
                    codec=codec,
                    minimum_mib=min(values),
                    median_mib=statistics.median(values),
                    maximum_mib=max(values),
                )
            )
    analysis = dict(
        configurations=len(rows),
        executions=len(samples),
        measured_executions=len(measured),
        repeats=sorted({row["repeats"] for row in rows}),
        median_span_pct=statistics.median(row["span"] for row in rows),
        maximum_span_pct=max(row["span"] for row in rows),
        widest_span_configuration=key(max(rows, key=lambda row: row["span"])),
        identical_historical_folds=sum(
            row["fold"] == old[key(row)]["fold"] for row in rows
        ),
        changed_frontier_groups=sum(change["changed"] for change in changes),
        frontier_groups=changes,
        memory_residuals=residuals,
    )
    (ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(
        json.dumps(
            {
                field: value
                for field, value in analysis.items()
                if field not in ("frontier_groups", "memory_residuals")
            },
            indent=2,
        )
    )
    return comparisons


def size_label(kib):
    return f"{kib // 1024} MiB" if kib >= 1024 and kib % 1024 == 0 else f"{kib:g} KiB"


def panel_title(fill, chunk):
    return f"{'12-bit random' if fill == 'rand' else 'Repetitive XOR'} · {size_label(chunk)} Zarr chunks"


def style_axes(ax):
    ax.grid(axis="y", color="#e3e8ec", linewidth=0.8)
    ax.grid(axis="x", color="#edf0f3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a3adb7")
    ax.tick_params(axis="both", which="both", length=0, pad=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))


def throughput_limit(points):
    return 2 * math.ceil(max(row["maximum"] for row in points) * 1.16 / 2)


def set_axes(ax, points, fill, ymax, zoom=True):
    if fill == "xor":
        ax.set_xscale("log")
        ax.set_xlim(1, 500)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
        ax.xaxis.set_minor_formatter(NullFormatter())
    elif zoom:
        if points[0]["chunk_kib"] == 256:
            ax.set_xlim(1.367, 1.39)
            ax.set_xticks([1.37, 1.375, 1.38, 1.385, 1.39])
        else:
            ax.set_xlim(1.47, 1.497)
            ax.set_xticks([1.47, 1.475, 1.48, 1.485, 1.49, 1.495])
    else:
        ax.set_xlim(1, 1.55)
        ax.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Compression fold →", labelpad=10)
    ax.set_ylabel("Input throughput (GiB/s) →", labelpad=10)
    style_axes(ax)


def add_panel_title(ax, title, scale):
    ax.text(
        -0.072,
        1.125,
        title,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
    )
    ax.text(
        -0.072,
        1.055,
        scale,
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED_COLOR,
        ha="left",
    )


def draw_points(ax, points, previous=False, faint=True):
    for codec, color in COLORS.items():
        candidates = [row for row in points if row["codec"] == codec]
        selected = frontier(candidates)
        if faint:
            for shuffle, marker in MARKERS.items():
                dominated = [
                    row
                    for row in candidates
                    if not row["per_codec_frontier"] and row["shuffle"] == shuffle
                ]
                ax.scatter(
                    [r["fold"] for r in dominated],
                    [r["speed"] for r in dominated],
                    color=color,
                    marker=marker,
                    s=22,
                    alpha=0.23,
                    linewidths=0,
                    zorder=1,
                )
        ax.plot(
            [r["fold"] for r in selected],
            [r["speed"] for r in selected],
            color=color,
            linestyle="--" if previous else "-",
            linewidth=1.8,
            zorder=2,
        )
        for row in selected:
            ax.errorbar(
                row["fold"],
                row["speed"],
                yerr=[[row["speed"] - row["minimum"]], [row["maximum"] - row["speed"]]],
                color=color,
                marker=MARKERS[row["shuffle"]],
                markerfacecolor="white" if previous else color,
                markersize=6,
                capsize=3,
                elinewidth=1,
                linestyle="none",
                zorder=4,
            )
    if faint:
        for row in points:
            if row["codec"] not in COLORS:
                ax.scatter(
                    row["fold"],
                    row["speed"],
                    marker="D",
                    facecolors="white",
                    edgecolors=COLORS[f"blosc-{row['codec']}"],
                    s=38,
                    linewidths=1.2,
                    zorder=3,
                )


def add_labels(ax, rows, other_rows=()):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    radius = renderer.points_to_pixels(6)
    padding = renderer.points_to_pixels(2)
    obstacles = []
    for row in [*rows, *other_rows]:
        x, low = ax.transData.transform((row["fold"], row["minimum"]))
        _, high = ax.transData.transform((row["fold"], row["maximum"]))
        obstacles.append(
            Bbox.from_extents(x - radius, low - radius, x + radius, high + radius)
        )
    lines = [
        line for line in ax.lines if line.get_linestyle() not in ("None", "", None)
    ]
    paths = [line.get_transform().transform_path(line.get_path()) for line in lines]
    offsets = sorted(
        (
            (dx, dy)
            for dx in (8, -8, 24, -24, 48, -48, 72, -72, 96, -96, 128, -128, 160, -160)
            for dy in (10, -10, 24, -24, 40, -40, 56, -56, 72, -72, 96, -96)
        ),
        key=lambda offset: (math.hypot(*offset), offset[1] < 0, offset[0] < 0),
    )
    notes, choices = [], []
    for row in rows:
        other_paths = [
            path
            for line, path in zip(lines, paths)
            if line.get_color() != COLORS[row["codec"]]
        ]
        note = ax.annotate(
            size_label(row["block_kib"]),
            (row["fold"], row["speed"]),
            xytext=(8, 10),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=COLORS[row["codec"]],
            zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.5),
            arrowprops=dict(
                arrowstyle="-",
                color=COLORS[row["codec"]],
                alpha=0.65,
                linewidth=0.65,
                shrinkA=2,
                shrinkB=5,
            ),
        )
        notes.append(note)
        available = []
        for dx, dy in offsets:
            note.set_position((dx, dy))
            note.set_ha("left" if dx > 0 else "right")
            note.update_positions(renderer)
            note.update_bbox_position_size(renderer)
            box = note.get_bbox_patch().get_window_extent(renderer).padded(padding)
            arrow = note.arrow_patch.get_transform().transform_path(
                note.arrow_patch.get_path()
            )
            if (
                ax.bbox.contains(box.x0, box.y0)
                and ax.bbox.contains(box.x1, box.y1)
                and not any(box.overlaps(old) for old in obstacles)
                and not any(path.intersects_bbox(box, filled=False) for path in paths)
                and not any(
                    arrow.intersects_path(path, filled=False) for path in other_paths
                )
            ):
                available.append(((dx, dy), box, arrow))
        if not available:
            raise ValueError(f"No clear label position for {key(row)}")
        choices.append(available)

    def compatible(candidate, placed):
        _, box, arrow = candidate
        return all(
            not box.overlaps(other_box)
            and not arrow.intersects_bbox(other_box, filled=False)
            and not other_arrow.intersects_bbox(box, filled=False)
            and not arrow.intersects_path(other_arrow, filled=False)
            for _, other_box, other_arrow in placed
        )

    selected = {}
    attempts = 0

    def place():
        nonlocal attempts
        attempts += 1
        if attempts > 10000:
            return False
        remaining = [
            (
                index,
                [
                    candidate
                    for candidate in available
                    if compatible(candidate, selected.values())
                ],
            )
            for index, available in enumerate(choices)
            if index not in selected
        ]
        if not remaining:
            return True
        index, available = min(remaining, key=lambda item: len(item[1]))
        for candidate in available:
            selected[index] = candidate
            if place():
                return True
            del selected[index]
        return False

    if not place():
        raise ValueError(f"No clear label layout for {key(rows[0])}")
    for index, note in enumerate(notes):
        (dx, dy), _, _ = selected[index]
        note.set_position((dx, dy))
        note.set_ha("left" if dx > 0 else "right")
        note.update_positions(renderer)
        note.update_bbox_position_size(renderer)


def legend_handles(comparison=False):
    handles = [
        Line2D([], [], color=color, linewidth=2, label=CODEC_NAMES[codec])
        for codec, color in COLORS.items()
    ]
    names = {"none": "no shuffle", "byte": "byte shuffle", "bit": "bitshuffle"}
    handles += [
        Line2D(
            [],
            [],
            color="#526373",
            marker=marker,
            markersize=6,
            linestyle="none",
            label=names[shuffle],
        )
        for shuffle, marker in MARKERS.items()
    ]
    if comparison:
        handles += [
            Line2D([], [], color="#526373", marker="o", label="L40"),
            Line2D(
                [],
                [],
                color="#526373",
                marker="o",
                markerfacecolor="white",
                linestyle="--",
                label="RTX 5070 Laptop",
            ),
        ]
    else:
        handles.append(
            Line2D(
                [],
                [],
                color="#526373",
                marker="D",
                markerfacecolor="white",
                linestyle="none",
                label="raw codec control",
            )
        )
    return handles


def figure_legend(fig, handles, y):
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.028, y),
        ncol=len(handles),
        frameon=False,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.7,
    )


def save_figure(fig, name):
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(ROOT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(ROOT / f"{name}.png", dpi=160)
    fig.savefig(ROOT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def make_plots(rows, previous):
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.labelweight": "bold",
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    repeats = rows[0]["repeats"]
    for name, zoom in (("pareto", True), ("pareto-all-candidates", False)):
        fig, axes = plt.subplots(2, 2, figsize=(15, 11.3))
        fig.subplots_adjust(
            left=0.0613,
            right=0.968,
            bottom=0.155,
            top=0.79,
            wspace=0.211,
            hspace=0.357,
        )
        fig.text(
            0.032,
            0.962,
            "GPU Blosc: throughput / compression Pareto frontiers",
            fontsize=20,
            fontweight="bold",
        )
        fig.text(
            0.032,
            0.936,
            "NVIDIA L40 · nvCOMP 5.3 · driver 580.126.20 · sweep 2026-09-06 UTC",
            fontsize=10.5,
            color="#506273",
        )
        fig.text(
            0.032,
            0.913,
            "Per-codec frontier across block sizes + all shuffle modes. Higher and farther right is better.",
            fontsize=10.5,
            color="#506273",
        )
        figure_legend(fig, legend_handles(), 0.888)
        fig.text(
            0.032,
            0.860,
            f"Labels: Blosc block size · bars: min–max of {repeats} runs · faint points: dominated candidates",
            fontsize=9.5,
            color=MUTED_COLOR,
        )
        ymax = throughput_limit(rows)
        for ax, (fill, chunk) in zip(axes.flat, PANELS):
            points = group(rows, fill, chunk)
            scale = (
                "log ratio axis"
                if fill == "xor"
                else "zoomed ratio axis"
                if zoom
                else "linear ratio axis"
            )
            add_panel_title(ax, panel_title(fill, chunk), scale)
            set_axes(ax, points, fill, ymax, zoom)
            draw_points(ax, points)
            add_labels(ax, [row for row in points if row["per_codec_frontier"]])
        notes = [
            "Frontiers compare tested Blosc settings within each codec; both median throughput and compression fold are maximized.",
            "Fold = padded chunk bytes / encoded bytes. Synthetic input and discard sink; GPU memory is not an objective here.",
            "Lines join measured settings. Bars show observed spread, not confidence intervals.",
            "Random panels zoom to the frontiers; off-scale candidates and raw controls are omitted."
            if zoom
            else "All measured candidates shown. Exact-median frontier membership can be sensitive to small differences.",
        ]
        for y, note in zip((0.079, 0.060, 0.041, 0.022), notes):
            fig.text(0.032, y, note, fontsize=9.5, color="#526373")
        save_figure(fig, name)

    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.subplots_adjust(
        left=0.0613,
        right=0.968,
        bottom=0.10,
        top=0.842,
        wspace=0.211,
        hspace=0.42,
    )
    fig.text(
        0.032,
        0.972,
        "Blosc frontiers: L40 and RTX 5070 Laptop",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.032,
        0.950,
        "L40: 2026-09-06 · RTX 5070 Laptop: 2026-09-05 · nvCOMP 5.3",
        fontsize=10.5,
        color="#506273",
    )
    fig.text(
        0.032,
        0.930,
        "Different source revisions and toolchains: a comparison of measured setups.",
        fontsize=10.5,
        color="#506273",
    )
    figure_legend(fig, legend_handles(comparison=True), 0.905)
    fig.text(
        0.032,
        0.880,
        "Labels: L40 block size · bars: observed min–max · L40: 5 runs · RTX 5070 Laptop: 3 runs",
        fontsize=9.5,
        color=MUTED_COLOR,
    )
    for axes_row, (fill, codec) in zip(
        axes, [(fill, codec) for fill in ("xor", "rand") for codec in COLORS]
    ):
        ymax = throughput_limit(
            [
                row
                for row in rows + previous
                if row["fill"] == fill and row["codec"] == codec
            ]
        )
        for ax, chunk in zip(axes_row, (256, 1024)):
            current, old = (
                group(rows, fill, chunk, codec),
                group(previous, fill, chunk, codec),
            )
            add_panel_title(
                ax,
                f"{panel_title(fill, chunk)} · {CODEC_NAMES[codec]}",
                "log ratio axis" if fill == "xor" else "zoomed ratio axis",
            )
            set_axes(ax, current + old, fill, ymax)
            draw_points(ax, old, previous=True, faint=False)
            draw_points(ax, current, faint=False)
            add_labels(
                ax,
                [row for row in current if row["per_codec_frontier"]],
                [row for row in old if row["per_codec_frontier"]],
            )
    for y, note in zip(
        (0.060, 0.043, 0.026),
        (
            "Frontiers include all tested blocks and shuffles within each codec. Lines join measured settings.",
            "Labels show L40 block sizes; the CSV and Markdown tables retain both sets of frontier identities.",
            "Bars show observed spread, not confidence intervals. This comparison does not isolate the effect of the GPU.",
        ),
    ):
        fig.text(0.032, y, note, fontsize=9.5, color="#526373")
    save_figure(fig, "comparison")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(left=0.0613, right=0.968, bottom=0.24, top=0.69, wspace=0.211)
    fig.text(
        0.032,
        0.94,
        "Block size and GPU memory",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.032,
        0.89,
        "NVIDIA L40 · XOR input · bitshuffle · fixed chunk and batch geometry · 6 GiB allocation budget",
        fontsize=10.5,
        color="#506273",
    )
    handles = [
        Line2D([], [], color=color, linewidth=2, label=CODEC_NAMES[codec])
        for codec, color in COLORS.items()
    ]
    handles += [
        Line2D([], [], color="#526373", marker="^", label="observed median delta"),
        Line2D(
            [],
            [],
            color="#526373",
            linestyle="--",
            label="explicit allocation estimate",
        ),
    ]
    figure_legend(fig, handles, 0.83)
    ymax = math.ceil(max(row["memory"] for row in rows))
    for ax, chunk in zip(axes, (256, 1024)):
        points = [
            row
            for row in group(rows, "xor", chunk)
            if row["codec"] in COLORS and row["shuffle"] == "bit"
        ]
        for codec, color in COLORS.items():
            selected = sorted(
                (row for row in points if row["codec"] == codec),
                key=lambda row: row["block_kib"],
            )
            blocks = [row["block_kib"] for row in selected]
            ax.plot(
                blocks,
                [row["memory"] for row in selected],
                color=color,
                marker=MARKERS["bit"],
                markersize=5,
                linewidth=1.8,
            )
            ax.plot(
                blocks,
                [row["estimate"] for row in selected],
                color=color,
                linestyle="--",
                linewidth=1.8,
            )
        blocks = sorted({row["block_kib"] for row in points})
        ax.set_xscale("log", base=2)
        ax.set_xticks(blocks, [size_label(block) for block in blocks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Blosc block size (log₂ scale)", labelpad=10)
        ax.set_ylabel("Device memory (GiB)", labelpad=10)
        ax.set_title(
            f"{size_label(chunk)} Zarr chunks · {144 if chunk == 256 else 288} MiB padded batch",
            loc="left",
            pad=16,
            fontsize=13,
            fontweight="bold",
        )
        style_axes(ax)
    fig.text(
        0.032,
        0.09,
        "Observed usage is a before/after free-memory delta while the stream is alive; it is not a sampled peak.",
        fontsize=9.5,
        color="#526373",
    )
    fig.text(
        0.032,
        0.045,
        "The baseline follows CUDA context creation. Runtime overhead is separate from explicit allocations.",
        fontsize=9.5,
        color="#526373",
    )
    save_figure(fig, "memory")


def make_html(rows, comparisons):
    pictures = []
    for name, title in (
        ("pareto", "L40 frontiers"),
        ("pareto-all-candidates", "All L40 candidates"),
        ("memory", "Device memory"),
        ("comparison", "Comparison with the RTX 5070 Laptop"),
    ):
        encoded = base64.b64encode((ROOT / f"{name}.svg").read_bytes()).decode()
        pictures.append(
            f'<details{" open" if name == "pareto" else ""}><summary>{html.escape(title)}</summary><img alt="{html.escape(title)}" src="data:image/svg+xml;base64,{encoded}"></details>'
        )
    table_rows = []
    paired = {key(row): row for row in comparisons}
    for row in rows:
        table_rows.append(
            dict(
                csv_row(row),
                rtx5070_throughput_median_gibs=paired[key(row)][
                    "rtx5070_throughput_median_gibs"
                ],
            )
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>L40 Blosc measurements</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1400px;padding:0 1rem;color:#202a35;background:#fff}
h1{font-size:1.8rem}p{max-width:100ch}img{width:100%;height:auto}details{margin:1.5rem 0}summary{cursor:pointer;font-weight:600}
label{display:inline-block;margin:0.3rem 1rem 0.3rem 0}select{font:inherit;padding:0.3rem}.scroll{overflow:auto;max-height:65vh}
table{border-collapse:collapse;font-size:0.85rem;width:100%;white-space:nowrap}th,td{text-align:right;padding:0.4rem 0.6rem;border-bottom:1px solid #ddd}
th{position:sticky;top:0;background:#edf3f7}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f7f9fb}
</style>
<h1>L40 Blosc measurements</h1>
<p>PR264, 200 configurations, one warmup and five measured repetitions. Throughput is original input GiB/s through the discard-sink pipeline; compression fold uses padded chunk bytes. Memory is a device-memory delta, not a sampled peak.</p>
<p>The archived RTX 5070 Laptop run used a different source revision, compiler, CUDA toolkit, and driver. These results compare measured setups and do not isolate hardware. Bars show observed min–max, not confidence intervals.</p>
PICTURES
<h2>Configuration measurements</h2>
<p>Frontier filters exclude raw controls. All shuffles compete within each input and chunk geometry. The memory frontier maximizes speed and ratio while minimizing observed memory across both Blosc codecs.</p>
<label>Input <select id="fill"><option value="">All</option><option value="rand">12-bit random</option><option value="xor">XOR</option></select></label>
<label>Chunk <select id="chunk_kib"><option value="">All</option><option value="256">256 KiB</option><option value="1024">1 MiB</option></select></label>
<label>Codec <select id="codec"><option value="">All</option><option>blosc-lz4</option><option>blosc-zstd</option><option>lz4</option><option>zstd</option></select></label>
<label>Shuffle <select id="shuffle"><option value="">All</option><option>none</option><option>byte</option><option>bit</option></select></label>
<label>Show <select id="frontier"><option value="">All candidates</option><option value="per_codec_frontier">Frontier within each codec</option><option value="cross_codec_frontier">Frontier across both codecs</option><option value="memory_frontier">Frontier including memory</option></select></label>
<label>Sort <select id="sort"><option value="throughput_median_gibs">L40 throughput</option><option value="compression_fold">Compression fold</option><option value="device_gib">Memory (lowest first)</option></select></label>
<p id="count" aria-live="polite"></p>
<div class="scroll"><table><thead><tr><th>Input</th><th>Chunk KiB</th><th>Codec</th><th>Shuffle</th><th>Block KiB</th><th>L40 GiB/s</th><th>L40 min–max</th><th>5070 GiB/s</th><th>Fold</th><th>Device GiB</th></tr></thead><tbody id="measurements"></tbody></table></div>
<script>
const rows = ROWS;
function render() {
  const fields = ["fill", "chunk_kib", "codec", "shuffle"];
  const frontier = document.getElementById("frontier").value;
  const sort = document.getElementById("sort").value;
  const visible = rows.filter(row => fields.every(field => !document.getElementById(field).value || String(row[field]) === document.getElementById(field).value) && (!frontier || row[frontier]));
  visible.sort((a, b) => (sort === "device_gib" ? 1 : -1) * (a[sort] - b[sort]));
  const body = document.getElementById("measurements");
  body.replaceChildren();
  for (const row of visible) {
    const tr = document.createElement("tr");
    const values = [row.fill, row.chunk_kib, row.codec, row.shuffle, row.block_kib || "—", row.throughput_median_gibs.toFixed(3), `${row.throughput_min_gibs.toFixed(3)}–${row.throughput_max_gibs.toFixed(3)}`, row.rtx5070_throughput_median_gibs.toFixed(3), row.compression_fold.toFixed(5), row.device_gib.toFixed(3)];
    for (const value of values) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    body.append(tr);
  }
  document.getElementById("count").textContent = `${visible.length} of ${rows.length} configurations`;
}
for (const select of document.querySelectorAll("select")) select.addEventListener("change", render);
render();
</script></html>
"""
    (ROOT / "pareto.html").write_text(
        document.replace("PICTURES", "\n".join(pictures)).replace(
            "ROWS", json.dumps(table_rows)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate retained measurements without regenerating outputs",
    )
    args = parser.parse_args()
    rows = read_summary(ROOT / "summary.csv")
    previous = read_summary(HISTORY / "summary.csv")
    provenance, samples = validate(rows, previous)
    mark_frontiers(rows)
    mark_frontiers(previous)
    if args.check:
        print(
            f"Validated {len(rows)} configurations and {len(samples)} executions; completed {provenance['finish_utc']}"
        )
        return
    comparisons = make_tables(rows, previous, samples)
    make_plots(rows, previous)
    make_html(rows, comparisons)


if __name__ == "__main__":
    main()
