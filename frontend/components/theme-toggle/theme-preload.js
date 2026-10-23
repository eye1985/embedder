// Classic blocking script, deliberately not a module: a module is deferred,
// and the theme has to be on the element before first paint or a stored choice
// flashes the other one. The button itself is theme-toggle.js.
(() => {
  try {
    const stored = localStorage.getItem("theme");
    if (stored === "light" || stored === "dark") {
      document.documentElement.dataset.theme = stored;
    }
  } catch {
    // Private browsing can throw on access; the OS preference still applies.
  }
})();
