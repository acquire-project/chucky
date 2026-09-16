export function machineChoices(container, choices, onChange) {
  container.replaceChildren(...choices.map(({id, label, caption}) => {
    const node = document.createElement("label"), checkbox = document.createElement("input");
    const copy = document.createElement("span"), title = document.createElement("strong"), detail = document.createElement("small");
    node.className = "system-choice";
    checkbox.type = "checkbox"; checkbox.value = id;
    title.textContent = label; detail.textContent = caption;
    copy.append(title, detail); node.append(checkbox, copy);
    checkbox.addEventListener("change", () => onChange([...container.querySelectorAll("input:checked")].map(input => input.value)));
    return node;
  }));
}

export function syncMachines(container, selected) {
  for (const checkbox of container.querySelectorAll("input")) checkbox.checked = selected.includes(checkbox.value);
}
