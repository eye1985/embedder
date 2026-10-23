import { el } from "../../lib/dom";

export function errorBox(message: string): HTMLParagraphElement {
  return el("p", "error-box", message);
}
