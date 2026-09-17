// Run metadata stays separate from machine descriptions and measurement values.
export function runDates(timestamps) {
  const dates = [...new Set(timestamps.filter(Boolean).map(value => new Date(value))
    .filter(date => Number.isFinite(date.getTime())).map(date => date.toISOString().slice(0, 10)))].sort();
  return dates.length > 1 ? `${dates[0]} – ${dates.at(-1)}` : dates[0] ?? "Not recorded";
}

export const machineName = experiment => experiment.machine_id ?? experiment.label;
export const runLabel = experiment => `${machineName(experiment)} · ${runDates([experiment.start_utc, experiment.finish_utc])} UTC`;
