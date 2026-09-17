import {machineFields} from "./machines.mjs";

const template = document.createElement("template");
template.innerHTML = `<span class="machine-name"></span>
  <span class="machine-description"></span><span class="machine-specs"></span>`;

// Light DOM keeps native labels, keyboard access and the site's theme available.
// All machine markup and its scoped styles are owned by this component.
class MachineSummary extends HTMLElement {
  set machine(value) {
    this.replaceChildren(template.content.cloneNode(true));
    this.dataset.machine = value.name;
    this.querySelector(".machine-name").textContent = value.name;
    const description = this.querySelector(".machine-description");
    description.textContent = value.description; description.hidden = !value.description;
    const specs = this.querySelector(".machine-specs");
    for (const [name, detail] of machineFields(value)) {
      const row = document.createElement("span"), label = document.createElement("span");
      row.className = "machine-spec"; label.className = "machine-spec-name";
      label.textContent = `${name} `; row.append(label, detail); specs.append(row);
    }
  }
}
customElements.define("machine-summary", MachineSummary);

export function machineSummary(machine, {showName = true} = {}) {
  const element = document.createElement("machine-summary");
  element.toggleAttribute("hide-name", !showName);
  element.machine = machine;
  return element;
}
