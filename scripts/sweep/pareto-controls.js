import {machineSummary} from "./machine-components.js";

const selections = new WeakMap();

export function machineChoices(container, choices, onChange) {
  const selection = {choices, selected: []}; selections.set(container, selection);
  container.replaceChildren(...choices.map(({id, machine, values = [id]}) => {
    const node = document.createElement("label"), checkbox = document.createElement("input");
    node.className = "system-choice";
    checkbox.type = "checkbox"; checkbox.value = id;
    node.append(checkbox, machineSummary(machine));
    checkbox.addEventListener("change", () => {
      const selected = new Set(selection.selected);
      for (const value of values) checkbox.checked ? selected.add(value) : selected.delete(value);
      onChange([...selected]);
    });
    return node;
  }));
}

export function syncMachines(container, selected) {
  const selection = selections.get(container); selection.selected = selected;
  for (const [i, checkbox] of [...container.querySelectorAll("input")].entries()) {
    const choice = selection.choices[i], values = choice.values ?? [choice.id];
    const count = values.filter(value => selected.includes(value)).length;
    checkbox.checked = count === values.length;
    checkbox.indeterminate = count > 0 && count < values.length;
  }
}
