import { el } from "../../lib/dom";
import { documentSourceHref, type Source } from "../../lib/endpoints";
import { bindDownload } from "../download/download";

export type Role = "user" | "assistant" | "error";

export interface Reply {
  /** What a streaming answer is written into. */
  body: HTMLElement;
  /** Names the documents the answer cited. An empty list renders nothing. */
  showSources: (sources: Source[]) => void;
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

// The documents the answer cited, which the server works out from the markers
// the model left in the text -- not everything retrieval turned up.
//
// One chip per document on a wrapping row, each carrying the filename the
// download arrives as. Pages live in the tooltip: ten chunks from one book is
// ten page numbers, which is wider than the answer and is not what the reader
// asked for.
// Clipped in the middle, never the end: the extension is what tells the
// reader the chip is a file, so it is the one part that has to survive.
function clip(name: string): string {
  if (name.length <= 34) return name;
  return `${name.slice(0, 22)}…${name.slice(-9)}`;
}

function sourceList(
  items: Source[],
  onError: (message: string) => void,
): HTMLElement {
  const node = el("div", "turn-sources");
  node.append(el("span", "turn-sources-label", "Sources"));

  for (const item of items) {
    const name = item.filename ?? item.title ?? "Untitled";

    // A document whose file is gone is still worth naming -- it was searched.
    // It just has nothing to download.
    let chip: HTMLElement;
    if (item.has_source) {
      const link = Object.assign(el("a", "source"), {
        href: documentSourceHref(item.document_id),
      });
      // `has_source` was true when the answer was generated, so the file can
      // still be gone by the time this is clicked. Downloaded rather than
      // followed so that lands in the transcript instead of replacing it.
      bindDownload(link, onError);
      chip = link;
    } else {
      chip = el("span", "source missing");
    }

    // The name is clipped when long, so the whole of it goes here too.
    chip.title = [
      name,
      item.pages.length > 0 ? `found on pages ${item.pages.join(", ")}` : "",
      item.has_source ? "click to download" : "file no longer stored",
    ]
      .filter((part) => part !== "")
      .join(" — ");

    chip.append(el("span", "source-name", clip(name)));
    node.append(chip);
  }

  return node;
}

// The message list. Owns its empty state and its scroll position.
export function transcript(node: HTMLElement): Transcript {
  function scrollToBottom() {
    node.scrollTop = node.scrollHeight;
  }

  // Named rather than written straight into the returned object: a failed
  // source download reports itself as a turn, so this has to reach itself.
  function add(role: Role, label: string, text = ""): Reply {
    node.querySelector(".empty")?.remove();
    const { node: item, body } = turn(role, label, text);
    node.append(item);
    scrollToBottom();
    return {
      body,
      showSources(sources) {
        if (sources.length === 0) return;
        item.append(
          sourceList(sources, (message) => {
            add("error", "Error", message);
            scrollToBottom();
          }),
        );
      },
      remove: () => item.remove(),
    };
  }

  return { add, scrollToBottom };
}
