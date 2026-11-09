import { need } from "../../lib/dom";

export interface Composer {
  unlock: () => void;
}

// The chat input: grows to fit, Enter sends, Shift+Enter breaks the line.
// `onSubmit` is handed the trimmed prompt and decides when to unlock.
export function composer(
  form: HTMLFormElement,
  onSubmit: (prompt: string) => void,
): Composer {
  const field = need(form.querySelector("textarea"), "composer textarea");
  const button = need(form.querySelector("button"), "composer button");
  const counter = need(
    form.querySelector<HTMLElement>(".composer-count"),
    "composer counter",
  );

  // Same unit as `maxlength`, UTF-16 code units, so the count matches what the
  // field allows.
  function count() {
    const limit = field.maxLength;
    counter.textContent = `${field.value.length} / ${limit}`;
    counter.classList.toggle("near", field.value.length >= limit * 0.9);
  }

  function autoGrow() {
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
    // The field grows to fit, so a bar is only wanted once max-height has
    // stopped it growing.
    field.style.overflowY =
      field.scrollHeight > field.clientHeight ? "auto" : "hidden";
  }

  field.addEventListener("input", () => {
    autoGrow();
    count();
  });
  autoGrow();
  count();

  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = field.value.trim();
    if (!prompt || button.disabled) return;

    field.value = "";
    autoGrow();
    count();
    button.disabled = true;
    onSubmit(prompt);
  });

  return {
    unlock() {
      button.disabled = false;
      field.focus();
    },
  };
}
