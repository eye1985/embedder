import { el, input } from "../../lib/dom";
import { count, formatDate } from "../../lib/format";
import { documentExportHref, documentSourceHref } from "../../lib/endpoints";
import type { DocumentOut, ModelOut } from "../../lib/endpoints";
import { chip, chipNote, chipRow } from "../chips/chips";
import { downloadLink } from "../download/download";
import { toggleSwitch } from "../switch/switch";

export interface DocActions {
  onSearchable: (searchable: boolean) => void;
  onSearchModel: (table: string) => void;
  /** Embed this document's stored chunks with one more model. Resolves false
   *  when it failed, so the button can come back. */
  onEmbed: (table: string) => Promise<boolean>;
  /** Resolves false when the delete failed, the one case where this card is
   *  still on screen afterwards. */
  onDelete: () => Promise<boolean>;
  /** A download that failed. The page owns where it is shown. */
  onError: (message: string) => void;
}

export interface DocContext {
  /** Models that have a vector table, which is the only kind that can be
   *  embedded into -- an inactive row has nothing to write to. */
  available: ModelOut[];
  /** Whether a job is already running. One at a time per user, so embedding
   *  is offered but refused while something else holds the slot. */
  busy: boolean;
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

// The models that could still embed this document: registered, with a vector
// table, and holding no vectors for it yet. This is what makes a document
// gain a model it was not uploaded with -- the chunks are already stored and
// are model-independent, so nothing is extracted again.
function addEmbedding(
  doc: DocumentOut,
  context: DocContext,
  onEmbed: (table: string) => Promise<boolean>,
): HTMLElement | null {
  const missing = context.available.filter(
    (model) => !doc.embedded[model.table_name],
  );
  if (!missing.length) return null;

  const row = el("div", "doc-add");
  row.append(el("span", "doc-add-label", "Add embedding"));

  for (const model of missing) {
    const button = el("button", "chip-button", model.model_name);
    button.type = "button";
    button.title = `${model.table_name} · ${model.dimensions}d`;
    button.disabled = context.busy;
    button.addEventListener("click", () => {
      if (!window.confirm(embedWarning(doc, model))) return;
      for (const other of row.querySelectorAll("button")) other.disabled = true;
      button.textContent = "Embedding…";
      void onEmbed(model.table_name).then((ok) => {
        if (!ok) {
          button.textContent = model.model_name;
          for (const other of row.querySelectorAll("button")) {
            other.disabled = false;
          }
        }
      });
    });
    row.append(button);
  }
  return row;
}

function embedWarning(doc: DocumentOut, model: ModelOut): string {
  const name = doc.title || doc.source_uri || "this document";
  return (
    `Embed "${name}" with ${model.model_name}?\n\n` +
    `${count(doc.chunks)} stored chunks will be embedded — the PDF is not ` +
    `read again. This spends against the embedding API, and runs as a job, ` +
    `so an upload cannot start until it finishes.\n\n` +
    `Which model answers questions about this document does not change; ` +
    `pick that separately once this is done.`
  );
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

export function docCard(
  doc: DocumentOut,
  actions: DocActions,
  context: DocContext,
): HTMLElement {
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

  const add = addEmbedding(doc, context, actions.onEmbed);
  if (add) item.append(add);

  // Chips and the row's own actions share a line, so a document reads as one
  // block rather than four stacked rows.
  const foot = el("div", "doc-foot");
  foot.append(modelChoice(doc, actions.onSearchModel));

  const docActions = el("div", "doc-actions");

  // Not gated on anything: /upload is the only thing that creates a document
  // and it always stores the PDF, which is deleted with the row it belongs to,
  // so a listed document has its file.
  docActions.append(
    downloadLink(
      "Download PDF",
      documentSourceHref(doc.document_id),
      actions.onError,
    ),
  );
  docActions.append(
    downloadLink(
      "Download JSON",
      documentExportHref(doc.document_id),
      actions.onError,
    ),
  );

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
