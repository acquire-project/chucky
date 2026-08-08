"""Pack a list of runs into columns so the pages stay small.

A sweep is thousands of runs with the same twenty-odd keys, so storing each run
as its own object spends most of its bytes repeating key names. One array per
key drops those. Text is worse than that: the same scenario and codec names, and
the same run ids, come back in every sweep, so all the text in a file shares one
table and the columns hold positions in it.

decode_runs is what the pages do on load, and pack checks every file against it
before report.py writes one.
"""

from __future__ import annotations

import math
from collections import Counter

# About what the benchmarks can actually resolve, and more than any page shows.
# Keeping the digits below this only adds bytes gzip cannot squeeze, since they
# look like noise to it.
SIGNIFICANT_DIGITS = 4


class StringTable:
    """Every distinct string in one output file, commonest first.

    Frequency order is what keeps the indices short: the handful of names that
    appear in every run land in the single digits.
    """

    def __init__(self, row_lists: list[list[dict]]) -> None:
        counts: Counter[str] = Counter()
        for rows in row_lists:
            for row in rows:
                for value in row.values():
                    if isinstance(value, str):
                        counts[value] += 1
        self.items = [text for text, _ in counts.most_common()]
        self.position = {text: i for i, text in enumerate(self.items)}


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


def encode_column(values: list, table: StringTable):
    # Only when every value present is text, so a column mixing text with
    # numbers stays as it is rather than looking a number up in the table.
    if {type(v) for v in values if v is not None} == {str}:
        return {"text": [None if v is None else table.position[v] for v in values]}
    return [round_float(v) for v in values]


def encode_rows(rows: list[dict], table: StringTable) -> dict:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return {
        "count": len(rows),
        "columns": {key: encode_column([r.get(key) for r in rows], table) for key in keys},
    }


def decode_runs(block: dict, strings: list[str]) -> list[dict]:
    """What the pages do on load."""
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


def pack(run_lists: list[list[dict]], omit: tuple[str, ...] = ()):
    """Columns for each list of runs, sharing one string table.

    Refuses to hand back a file the pages would unpack differently, so a broken
    encoding stops the build instead of reaching a page.
    """
    row_lists = [[flatten(run, omit) for run in runs] for runs in run_lists]
    table = StringTable(row_lists)
    blocks = [encode_rows(rows, table) for rows in row_lists]
    for rows, block in zip(row_lists, blocks):
        check(rows, block, table.items)
    return table.items, blocks


def check(rows: list[dict], block: dict, strings: list[str]) -> None:
    for index, (row, restored) in enumerate(zip(rows, decode_runs(block, strings))):
        want = {k: round_float(v) for k, v in row.items() if v is not None}
        got = flatten(restored)
        if got != want:
            differing = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
            raise ValueError(f"run {index} ({row.get('id', 'no id')}) changed: {differing}")
