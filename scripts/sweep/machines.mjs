import {fetchJson} from "./decode.js";

export async function loadMachines() {
  const data = await fetchJson("data/machines.json");
  if (data.version !== 1 || !Array.isArray(data.machines)) throw new Error("Unsupported machine catalog. Rebuild the report.");
  return new Map(data.machines.map(machine => [machine.name, machine]));
}

export function machineDescription(catalog, name) {
  return catalog.get(name) ?? {name, description: "Machine is not in the registry", specs: {}};
}

export function machineFields(machine) {
  const specs = machine.specs ?? {};
  const labels = {gpu: "GPU", cpu: "CPU", storage: "Storage", memory: "Memory", os: "OS"};
  const keys = [...new Set(["gpu", "cpu", "storage", ...Object.keys(specs).sort()])];
  return keys.map(key => [labels[key] ?? key, specs[key] || "Not recorded"]);
}

export function machineGroups(experiments) {
  const groups = new Map();
  for (const experiment of experiments) {
    const id = experiment.machine_id ?? experiment.id;
    if (!groups.has(id)) groups.set(id, {id, values: []});
    groups.get(id).values.push(experiment.id);
  }
  return [...groups.values()].sort((a, b) => a.id.localeCompare(b.id, "en"));
}
