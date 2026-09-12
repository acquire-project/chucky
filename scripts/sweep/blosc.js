export function bloscBlockKey(run) {
  return run.codec.startsWith("blosc-") ? String(run.blosc_block_bytes ?? "unknown") : "";
}

export function bloscBlockLabel(key) {
  if (!key) return "";
  if (key === "unknown") return "unknown (not recorded)";
  const bytes = Number(key);
  return bytes % 1024 === 0 ? `${bytes / 1024} KiB` : `${bytes} B`;
}
