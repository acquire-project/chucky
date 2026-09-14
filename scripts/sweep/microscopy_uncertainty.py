"""Describe variation by resampling matched measurement rounds."""
from __future__ import annotations

import random
import statistics


def interval(values):
    ordered = sorted(values)

    def percentile(p):
        position = p * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"lower": percentile(0.025), "upper": percentile(0.975)}


def summarize_rounds(rows, *, draws=2000, seed=20260913):
    if type(draws) is not int or draws < 1:
        raise ValueError("Resampling requires a positive draw count")
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    for group in groups.values():
        samples = [{sample["round"]: sample for sample in row["samples"]} for row in group]
        rounds = sorted(samples[0])
        if len(rounds) < 6 or any(sorted(values) != rounds for values in samples):
            continue
        rates, folds = [[] for _ in group], [[] for _ in group]
        frequency = [0] * len(group)
        rng = random.Random(seed)
        for _ in range(draws):
            selected = rng.choices(rounds, k=len(rounds))
            points = []
            for index, values in enumerate(samples):
                observed = [values[number] for number in selected]
                rate = statistics.median(sample["throughput_gibs"] for sample in observed)
                fold = sum(sample["logical_input_bytes"] for sample in observed) / sum(
                    sample["output_bytes"] for sample in observed)
                rates[index].append(rate)
                folds[index].append(fold)
                points.append((rate, fold))
            for index, (rate, fold) in enumerate(points):
                if not any(other_rate >= rate and other_fold >= fold
                           and (other_rate > rate or other_fold > fold) for other_rate, other_fold in points):
                    frequency[index] += 1
        for index, row in enumerate(group):
            row["uncertainty"] = {
                "method": "paired-round-bootstrap", "rounds": len(rounds), "draws": draws,
                "interval_percent": 95, "throughput": interval(rates[index]),
                "compression_fold": interval(folds[index]),
                "frontier_frequency": frequency[index] / draws,
                "configurations": len(group),
                "scope": "All selected settings in this study/input/backend/sink condition",
            }
