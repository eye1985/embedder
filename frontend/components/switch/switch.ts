import { el, input } from "../../lib/dom";

interface SwitchOptions {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

// A labelled checkbox. Disabled the moment it is used: the caller re-renders
// from the server, so nothing on screen is left claiming a state that was not
// accepted.
export function toggleSwitch({
  label,
  checked,
  onChange,
}: SwitchOptions): HTMLLabelElement {
  const node = el("label", "switch");

  const box = input("checkbox", { checked });
  box.addEventListener("change", () => {
    box.disabled = true;
    onChange(box.checked);
  });

  node.append(box);
  node.append(el("span", null, label));
  return node;
}
