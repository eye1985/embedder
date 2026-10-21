// Runs before first paint so a stored theme does not flash the other one.
(() => {
  const root = document.documentElement;
  const KEY = "theme";

  let stored = null;
  try {
    stored = localStorage.getItem(KEY);
  } catch {
    // Private browsing can throw on access; the OS preference still applies.
  }
  if (stored === "light" || stored === "dark") root.dataset.theme = stored;

  function resolved() {
    if (root.dataset.theme) return root.dataset.theme;
    return matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  // Built here rather than in each page's markup: the control is useless
  // without this script, and the icons are the same on all three pages.
  const ICONS = `
    <svg class="icon-sun" viewBox="0 0 20 20" fill="none" stroke="currentColor"
         stroke-width="1.4" stroke-linecap="round" aria-hidden="true">
      <circle cx="10" cy="10" r="3.6" />
      <path d="M10 1.6v1.8M10 16.6v1.8M1.6 10h1.8M16.6 10h1.8M4.1 4.1l1.3 1.3M14.6 14.6l1.3 1.3M15.9 4.1l-1.3 1.3M5.4 14.6l-1.3 1.3" />
    </svg>
    <svg class="icon-moon" viewBox="0 0 20 20" fill="none" stroke="currentColor"
         stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M16.2 12.6A7 7 0 0 1 7.4 3.8a7 7 0 1 0 8.8 8.8Z" />
    </svg>`;

  addEventListener("DOMContentLoaded", () => {
    const masthead = document.querySelector(".masthead");
    if (!masthead) return;

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "theme-toggle";
    toggle.setAttribute("aria-label", "Switch colour theme");
    toggle.innerHTML = ICONS;

    toggle.addEventListener("click", () => {
      const next = resolved() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem(KEY, next);
      } catch {
        // Not persisted, but the page still switches.
      }
    });

    masthead.append(toggle);
  });
})();
