import { el } from "../../lib/dom";

export function emptyState(title: string, note: string): HTMLDivElement {
  const box = el("div", "empty");
  box.append(el("p", "empty-title", title));
  box.append(el("p", "empty-note", note));
  return box;
}
