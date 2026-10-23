import { el, input } from "../../lib/dom";
import { count, formatDate } from "../../lib/format";
import { documentExportHref } from "../../lib/endpoints";
import type { DocumentOut } from "../../lib/endpoints";
import { chip, chipNote, chipRow } from "../chips/chips";
import { toggleSwitch } from "../switch/switch";

export interface DocActions {
  onSearchable: (searchable: boolean) => void;
  onSearchModel: (table: string) => void;
  /** Resolves false when the delete failed, the one case where this card is
   *  still on screen afterwards. */
  onDelete: () => Promise<boolean>;
}

// One radio per model holding vectors for this document: which one it is
// retrieved through. Models that embedded nothing are left out -- there is
// nothing to choose there, and the chunk counts are already in `embedded`.
//
// A document with a single model still shows it: seeing what will answer for
// it matters more than hiding a control with one option. That radio is only
// locked once it is the one chosen -- a document left with no model after its
// vectors were deleted has to be selectable, or it could never be searched
// again.
//
// Nothing here is live while the document is switched off: which model would
// answer for it is not a question until it is searched at all.
function modelChoice(
  doc: DocumentOut,
  onChoose: (table: string) => void,
): HTMLElement {
  const row = chipRow();
  const tables = Object.entries(doc.embedded).filter(([, n]) => n);

  if (!tables.length) {
    row.append(chipNote("no embeddings — not searchable"));
    return row;
  }

  for (const [table, n] of tables) {
    const active = table === doc.search_model;
    const control = input("radio", {
      name: `search-model-${doc.document_id}`,
      value: table,
      checked: active,
      disabled: !doc.searchable || (tables.length === 1 && active),
    });
    control.addEventListener("change", () => {
      for (const button of row.querySelectorAll("input")) {
        button.disabled = true;
      }
      onChoose(table);
    });
    row.append(chip({ name: table, count: count(n), active, control }));
  }
  return row;
}

function deleteWarning(doc: DocumentOut): string {
  const name = doc.title || doc.source_uri || "this document";
  const vectors = Object.values(doc.embedded).reduce((a, b) => a + b, 0);
  return (
    `Delete "${name}"?\n\n` +
    `${count(doc.chunks)} chunks and ${count(vectors)} embeddings will be ` +
    `removed, along with the uploaded PDF. This cannot be undone.`
  );
}

export function docCard(doc: DocumentOut, actions: DocActions): HTMLElement {
  const item = el("article", "doc");

  const head = el("div", "doc-head");
  head.append(el("h2", "doc-title", doc.title || "Untitled"));
  head.append(
    toggleSwitch({
      label: "Searchable",
      checked: doc.searchable,
      onChange: actions.onSearchable,
    }),
  );
  item.append(head);

  const meta = el("p", "doc-meta");
  meta.append(el("span", null, doc.source_uri || "no source"));
  meta.append(el("span", null, `${count(doc.chunks)} chunks`));
  meta.append(el("span", null, formatDate(doc.created_at)));
  item.append(meta);

  // Chips and the row's own actions share a line, so a document reads as one
  // block rather than four stacked rows.
  const foot = el("div", "doc-foot");
  foot.append(modelChoice(doc, actions.onSearchModel));

  const docActions = el("div", "doc-actions");
  const link = el("a", "button small", "Download JSON");
  link.href = documentExportHref(doc.document_id);
  docActions.append(link);

  const remove = el("button", "button small danger", "Delete");
  remove.type = "button";
  remove.addEventListener("click", () => {
    if (!window.confirm(deleteWarning(doc))) return;
    remove.disabled = true;
    remove.textContent = "Deleting…";
    void actions.onDelete().then((ok) => {
      if (!ok) {
        remove.disabled = false;
        remove.textContent = "Delete";
      }
    });
  });
  docActions.append(remove);

  foot.append(docActions);
  item.append(foot);

  if (!doc.searchable) item.classList.add("off");
  return item;
}
