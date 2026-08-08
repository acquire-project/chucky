"""Pack a list of runs into columns so the pages stay small.

A sweep is thousands of runs with the same twenty-odd keys, so storing each run
as its own object spends most of its bytes repeating key names. One array per
key drops those. Text is worse than that: the same scenario and codec names, and
the same run ids, come back in every sweep, so all the text in a file shares one
table and the columns hold positions in it.

The pages undo this at load time, so everything downstream still sees a plain
list of run objects.
"""

from __future__ import annotations

import math
from collections import Counter

# The explorer draws from the config fields, never from the recorded id, and the
# id is the longest string in a sweep. It is kept in the overview, whose movers
# panel matches runs between sweeps by it.
EXPLORER_OMITS = ("id",)

# About what the benchmarks can actually resolve, and more than any page shows.
# Keeping the digits below this only adds bytes gzip cannot squeeze, since they
# look like noise to it. It does move the movers panel, which compares numbers
# far finer than four digits — but those comparisons were reporting measurement
# noise, so losing them is the point rather than a cost.
SIGNIFICANT_DIGITS = 4


class StringTable:
    """Every distinct string in one output file, commonest first.

    Frequency order is what keeps the indices short: the handful of names that
    appear in every run land in the single digits.
    """

    def __init__(self, run_lists: list[list[dict]], omit: tuple[str, ...] = ()) -> None:
        counts: Counter[str] = Counter()
        for runs in run_lists:
            for run in runs:
                for value in flatten(run, omit).values():
                    if isinstance(value, str):
                        counts[value] += 1
        self.items = [text for text, _ in counts.most_common()]
        self._position = {text: i for i, text in enumerate(self.items)}

    def index(self, value: str) -> int:
        return self._position[value]


def round_float(value):
    if isinstance(value, float) and math.isfinite(value):
        return float(f"%.{SIGNIFICANT_DIGITS}g" % value)
    return value


def flatten(run: dict, omit: tuple[str, ...] = ()) -> dict:
    """Nested groups such as stalls and stages become dotted keys."""
    out = {}
    for key, value in run.items():
        if key in omit:
            continue
        if isinstance(value, dict):
            for sub, inner in value.items():
                if isinstance(inner, dict):
                    for leaf, number in inner.items():
                        out[f"{key}.{sub}.{leaf}"] = number
                else:
                    out[f"{key}.{sub}"] = inner
        else:
            out[key] = value
    return out


def encode_column(values: list, strings: StringTable):
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, str) for v in present):
        return {"text": [None if v is None else strings.index(v) for v in values]}
    return [round_float(v) for v in values]


def decode_runs(block: dict, strings: list[str]) -> list[dict]:
    """What the pages do on load. Here so report.py can check its own output."""
    runs: list[dict] = [{} for _ in range(block["count"])]
    for key, column in block["columns"].items():
        text = column["text"] if isinstance(column, dict) else None
        values = text if text is not None else column
        *path, leaf = key.split(".")
        for run, packed in zip(runs, values):
            if packed is None:
                continue
            node = run
            for step in path:
                node = node.setdefault(step, {})
            node[leaf] = strings[packed] if text is not None else packed
    return runs


def rounded(run: dict, omit: tuple[str, ...] = ()) -> dict:
    """One run as the encoder will store it, for comparing against a decode.

    An empty group is left out because flatten gives it no column, so unpacking
    cannot bring it back and comparing against it would fail a sound encoding.
    """
    out = {}
    for key, value in run.items():
        if key in omit:
            continue
        if isinstance(value, dict):
            group = rounded(value)
            if group:
                out[key] = group
        elif value is not None:
            out[key] = round_float(value)
    return out


def encode_runs(runs: list[dict], strings: StringTable, omit: tuple[str, ...] = ()) -> dict:
    flat = [flatten(r, omit) for r in runs]
    keys: list[str] = []
    for run in flat:
        for key in run:
            if key not in keys:
                keys.append(key)
    return {
        "count": len(flat),
        "columns": {
            key: encode_column([r.get(key) for r in flat], strings) for key in keys
        },
    }
