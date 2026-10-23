export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string | null,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

type InputProps = Partial<
  Pick<HTMLInputElement, "name" | "value" | "checked" | "disabled">
>;

export function input(type: string, props: InputProps = {}): HTMLInputElement {
  const node = document.createElement("input");
  node.type = type;
  Object.assign(node, props);
  return node;
}

// A page's markup is a contract with its script. A missing id is a template
// bug, so it fails here naming the id, rather than as a null somewhere later.
export function must(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (node === null) throw new Error(`No #${id} in the page markup.`);
  return node;
}

export function mustBe<T extends HTMLElement>(
  id: string,
  type: { new (): T },
): T {
  const node = must(id);
  if (!(node instanceof type)) {
    throw new Error(`#${id} is a <${node.tagName.toLowerCase()}>, unexpectedly.`);
  }
  return node;
}

export function need<T>(value: T | null | undefined, what: string): T {
  if (value === null || value === undefined) throw new Error(`No ${what}.`);
  return value;
}
