import { el } from "../../lib/dom";

// Also read by theme-preload.js, which applies the stored value before first
// paint. Both must agree on this key.
const KEY = "theme";

// Built here rather than in the page shell: the control is useless without
// this script, and the icons are the same on every page.
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

type Theme = "light" | "dark";

function resolved(): Theme {
  const chosen = document.documentElement.dataset.theme;
  if (chosen === "light" || chosen === "dark") return chosen;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function themeToggle(): HTMLButtonElement {
  const toggle = el("button", "theme-toggle");
  toggle.type = "button";
  toggle.setAttribute("aria-label", "Switch colour theme");
  toggle.innerHTML = ICONS;

  toggle.addEventListener("click", () => {
    const next: Theme = resolved() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // Not persisted, but the page still switches.
    }
  });

  return toggle;
}

// Page modules are deferred, so the masthead is already parsed by now.
export function mountThemeToggle(): void {
  document.querySelector(".masthead")?.append(themeToggle());
}
