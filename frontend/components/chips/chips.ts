import { el } from "../../lib/dom";

export function chipRow(): HTMLDivElement {
  return el("div", "chips");
}

interface ChipOptions {
  name: string;
  count: string;
  active?: boolean;
  zero?: boolean;
  /** An input the chip wraps, which makes the chip a label and its whole
   *  area the hit target. */
  control?: HTMLInputElement;
}

export function chip({
  name,
  count,
  active = false,
  zero = false,
  control,
}: ChipOptions): HTMLElement {
  const classes = ["chip"];
  if (zero) classes.push("zero");
  if (active) classes.push("active");

  const node = control
    ? el("label", classes.join(" "))
    : el("span", classes.join(" "));
  if (control) node.append(control);
  node.append(el("span", "chip-name", name));
  node.append(el("span", "chip-count", count));
  return node;
}

// A chip that says something rather than counting something.
export function chipNote(text: string): HTMLSpanElement {
  return el("span", "chip zero", text);
}
