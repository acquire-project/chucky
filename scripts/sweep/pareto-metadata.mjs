// Run metadata and host descriptions stay separate from measurement values.
export function hardwareDetails(records) {
  const recorded = value => value && value !== "unknown";
  return [["GPU", "gpu"], ["CPU", "cpu"], ["Storage", "storage"]].map(([label, key]) => {
    const values = records.map(({hardware = {}, specs = {}}) => {
      const value = key === "cpu" ? hardware.cpu || hardware.cpu_models?.join(" / ") : hardware[key];
      return recorded(value) ? value : specs[key];
    }).filter(recorded);
    return [label, [...new Set(values)].join(" / ") || "Not recorded"];
  });
}

export function runDates(timestamps) {
  const dates = [...new Set(timestamps.filter(Boolean).map(value => new Date(value))
    .filter(date => Number.isFinite(date.getTime())).map(date => date.toISOString().slice(0, 10)))].sort();
  return dates.length > 1 ? `${dates[0]} – ${dates.at(-1)}` : dates[0] ?? "Not recorded";
}
