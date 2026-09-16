export function machineChoices(container, choices, onChange) {
  container.replaceChildren(...choices.map(({id, label, details}) => {
    const node = document.createElement("label"), checkbox = document.createElement("input");
    const copy = document.createElement("span"), title = document.createElement("strong");
    node.className = "system-choice";
    checkbox.type = "checkbox"; checkbox.value = id;
    title.textContent = label; copy.append(title);
    for (const [name, value] of details) {
      const detail = document.createElement("small"), key = document.createElement("span");
      key.className = "machine-spec-name"; key.textContent = `${name} `;
      detail.append(key, value); copy.append(detail);
    }
    node.append(checkbox, copy);
    checkbox.addEventListener("change", () => onChange([...container.querySelectorAll("input:checked")].map(input => input.value)));
    return node;
  }));
}

export function syncMachines(container, selected) {
  for (const checkbox of container.querySelectorAll("input")) checkbox.checked = selected.includes(checkbox.value);
}
