export function bloscBlockKey(run) {
  return run.codec.startsWith("blosc-") ? String(run.blosc_block_bytes ?? "unknown") : "";
}

export function bloscBlockLabel(key) {
  if (!key) return "";
  if (key === "unknown") return "unknown (not recorded)";
  const bytes = Number(key);
  return bytes % 1024 === 0 ? `${bytes / 1024} KiB` : `${bytes} B`;
}

export function bloscBlockChoices(runs, codec) {
  if (!codec?.startsWith("blosc-")) return [];
  return [...new Set(runs.filter(r => r.codec === codec).map(bloscBlockKey))]
    .sort((a, b) => a === b ? 0 : a === "unknown" ? 1 : b === "unknown" ? -1 : Number(a) - Number(b));
}

export function matchesBloscBlock(run, selected) {
  return !run.codec.startsWith("blosc-") || bloscBlockKey(run) === selected;
}
