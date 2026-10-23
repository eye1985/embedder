import { el } from "../../lib/dom";
import { count } from "../../lib/format";
import { documentExportHref, documentsPageHref } from "../../lib/endpoints";
import type { ImportResult } from "../../lib/endpoints";
import { chip, chipRow } from "../chips/chips";
import { errorBox } from "../error-box/error-box";

export interface UploadStatusView {
  progress: (filename: string, elapsed: number | null) => void;
  report: (result: ImportResult) => void;
  error: (message: string) => void;
  clear: () => void;
}

// The outcome panel under the upload form: running, finished, or failed.
// Render only -- the polling that decides which of these to show lives on the
// page, because an upload outlives the page that started it.
export function uploadStatus(node: HTMLElement): UploadStatusView {
  return {
    progress(filename, elapsed) {
      const card = el("article", "card progress");
      card.append(el("h2", "card-title", `Embedding ${filename}…`));

      const meta = el("p", "card-meta");
      meta.append(
        el(
          "span",
          null,
          elapsed ? `${elapsed.toFixed(0)}s elapsed` : "starting",
        ),
      );
      card.append(meta);

      card.append(
        el(
          "p",
          "note",
          "This runs on the server. You can reload or close this page — the " +
            "upload keeps going and will show up under Documents.",
        ),
      );
      node.replaceChildren(card);
    },

    report(result) {
      const card = el("article", "card");
      card.append(el("h2", "card-title", result.source_uri));

      const meta = el("p", "card-meta");
      meta.append(
        el(
          "span",
          null,
          result.reused_existing_document
            ? "Already ingested — chunks reused"
            : "Ingested",
        ),
      );
      meta.append(el("span", null, `${count(result.chunks)} chunks`));
      card.append(meta);

      // `embedded` counts vectors written by THIS call, so a zero means the
      // model already held every chunk -- not that anything failed. The marked
      // chip is the model the document is now searched with.
      const chips = chipRow();
      for (const [table, n] of Object.entries(result.embedded)) {
        chips.append(
          chip({
            name: table,
            count: `+${count(n)}`,
            active: table === result.search_model,
            zero: !n,
          }),
        );
      }
      card.append(chips);

      const actions = el("div", "card-actions");
      const link = el("a", "button small", "Download embeddings");
      link.href = documentExportHref(result.document_id);
      actions.append(link);
      const all = el("a", "link", "See all documents");
      all.href = documentsPageHref;
      actions.append(all);
      card.append(actions);

      node.replaceChildren(card);
    },

    error(message) {
      node.replaceChildren(errorBox(message));
    },

    clear() {
      node.replaceChildren();
    },
  };
}
