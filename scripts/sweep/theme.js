// Light or dark, for both pages. Loaded before the body so a stored choice is
// applied before the page paints rather than after its data arrives.

function storedTheme() {
  // Reading storage throws in some private-browsing contexts, and this runs
  // before anything is drawn, so a preference must never be fatal.
  try {
    const stored = localStorage.getItem("bench-theme");
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    return null;
  }
}

function currentTheme() {
  return storedTheme() ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
}

/** Wire the title bar's button. onChange redraws whatever the page has drawn. */
function wireThemeToggle(onChange) {
  const button = document.getElementById("theme-toggle");
  const paint = () => { button.textContent = currentTheme() === "dark" ? "Light" : "Dark"; };
  button.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("bench-theme", next);
    } catch {
      // The choice still applies to this page; it just will not be remembered.
    }
    applyTheme(next);
    paint();
    onChange();
  });
  paint();
}

applyTheme(currentTheme());
