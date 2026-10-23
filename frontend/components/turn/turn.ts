import { el } from "../../lib/dom";

export type Role = "user" | "assistant" | "error";

export interface Reply {
  /** What a streaming answer is written into. */
  body: HTMLElement;
  /** Takes the whole turn back out, for a reply that failed. */
  remove: () => void;
}

export interface Transcript {
  add: (role: Role, label: string, text?: string) => Reply;
  scrollToBottom: () => void;
}

function turn(role: Role, label: string, text: string) {
  const node = el("div", `turn ${role}`);
  node.append(el("div", "turn-role", label));
  const body = el("div", "turn-body", text);
  node.append(body);
  return { node, body };
}

// The message list. Owns its empty state and its scroll position.
export function transcript(node: HTMLElement): Transcript {
  function scrollToBottom() {
    node.scrollTop = node.scrollHeight;
  }

  return {
    add(role, label, text = "") {
      node.querySelector(".empty")?.remove();
      const { node: item, body } = turn(role, label, text);
      node.append(item);
      scrollToBottom();
      return { body, remove: () => item.remove() };
    },
    scrollToBottom,
  };
}
