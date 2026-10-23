import { el, input } from "../../lib/dom";
import { listModels } from "../../lib/endpoints";
import { errorBox } from "../error-box/error-box";

interface ModelPickerOptions {
  /** Container for the "embed with" checkboxes. */
  options: HTMLElement;
  /** The "search this document with" section, hidden under two models. */
  section: HTMLElement;
  /** Container for that section's radios. */
  choices: HTMLElement;
  onChange: () => void;
}

export interface ModelPicker {
  load: () => Promise<void>;
  chosenModels: () => string[];
  searchModel: () => string | null;
  setBusy: (on: boolean) => void;
}

// Which models embed the upload, and which of them the document is then
// retrieved through. `onChange` fires whenever either selection moves, so the
// page can re-decide whether submitting is allowed.
export function modelPicker({
  options,
  section,
  choices,
  onChange,
}: ModelPickerOptions): ModelPicker {
  let boxes: { box: HTMLInputElement; available: boolean; label: string }[] = [];
  // Rebuilt whenever the checked set changes, so it can only ever offer a
  // model that is actually being embedded.
  let buttons: HTMLInputElement[] = [];
  let busy = false;

  function chosenModels(): string[] {
    return boxes.filter(({ box }) => box.checked).map(({ box }) => box.value);
  }

  function searchModel(): string | null {
    return buttons.find((button) => button.checked)?.value ?? null;
  }

  // One model embedded means no choice to make, so the radios only appear from
  // two up. The current pick survives a rebuild if it is still being embedded.
  function refreshChoice() {
    const tables = chosenModels();
    const keep = searchModel();

    section.hidden = tables.length < 2;
    choices.replaceChildren();
    buttons = [];

    if (section.hidden) {
      onChange();
      return;
    }

    for (const { box, label } of boxes) {
      if (!box.checked) continue;

      const option = el("label", "radio");
      const button = input("radio", {
        name: "search-model",
        value: box.value,
        checked: box.value === keep,
        disabled: busy,
      });
      button.addEventListener("change", onChange);
      option.append(button);
      option.append(el("span", "radio-name", label));
      choices.append(option);
      buttons.push(button);
    }

    onChange();
  }

  // Fetched rather than hardcoded: which models exist is a database question,
  // and a row without a vector table cannot be embedded into.
  async function load(): Promise<void> {
    try {
      const models = await listModels();

      options.replaceChildren();
      boxes = [];

      for (const model of models) {
        const label = `${model.provider} · ${model.model_name}`;
        const option = el(
          "label",
          model.available ? "model" : "model unavailable",
        );

        const box = input("checkbox", {
          value: model.table_name,
          checked: model.is_upload_default,
          disabled: !model.available,
        });
        box.addEventListener("change", refreshChoice);
        option.append(box);

        const text = el("span", "model-text");
        text.append(el("span", "model-name", label));
        text.append(
          el(
            "span",
            "model-meta",
            model.available
              ? `${model.table_name} · ${model.dimensions}d${model.is_upload_default ? " · default" : ""}`
              : `${model.table_name} · no vector table — run python -m db.setup`,
          ),
        );
        option.append(text);

        options.append(option);
        boxes.push({ box, available: model.available, label });
      }
      refreshChoice();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Could not load models.";
      options.replaceChildren(errorBox(message));
    }
  }

  return {
    load,
    chosenModels,
    searchModel,
    setBusy(on: boolean) {
      busy = on;
      for (const { box, available } of boxes) box.disabled = on || !available;
      for (const button of buttons) button.disabled = on;
    },
  };
}
