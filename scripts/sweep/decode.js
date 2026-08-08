// Reading side of the packing report.py applies. Both pages import this, so the
// format has one implementation here and one in columnar.py, which checks
// itself against this shape before writing a file.

export async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

/** Columns back to one object per run. */
export function decodeRuns(block, strings) {
  const runs = Array.from({length: block.count}, () => ({}));
  for (const [key, column] of Object.entries(block.columns)) {
    const text = column.text;
    const values = text || column;
    const path = key.split(".");
    const leaf = path.pop();
    for (let i = 0; i < block.count; i++) {
      const packed = values[i];
      if (packed == null) continue;
      let node = runs[i];
      for (const step of path) node = node[step] ??= {};
      node[leaf] = text ? strings[packed] : packed;
    }
  }
  return runs;
}
