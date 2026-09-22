const list = document.getElementById("list");
const summary = document.getElementById("summary");
const downloadAll = document.getElementById("download-all");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? iso : date.toLocaleString();
}

// `embedded` is keyed by vector table name, one entry per registered model.
// Models with no vectors are still shown, dimmed -- "this model has not
// embedded this document" is the thing a reader is usually checking for.
function modelChips(embedded) {
  const row = el("div", "chips");
  for (const [table, count] of Object.entries(embedded)) {
    const chip = el("span", count ? "chip" : "chip zero");
    chip.append(el("span", "chip-name", table));
    chip.append(el("span", "chip-count", count.toLocaleString()));
    row.append(chip);
  }
  return row;
}

async function removeDocument(doc, button) {
  const name = doc.title || doc.source_uri || "this document";
  const vectors = Object.values(doc.embedded).reduce((a, b) => a + b, 0);
  const warning =
    `Delete "${name}"?\n\n` +
    `${doc.chunks.toLocaleString()} chunks and ` +
    `${vectors.toLocaleString()} embeddings will be removed, ` +
    `along with the uploaded PDF. This cannot be undone.`;
  if (!window.confirm(warning)) return;

  button.disabled = true;
  button.textContent = "Deleting…";

  try {
    const response = await fetch(`/documents/${doc.document_id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(
        (payload && payload.detail) || `Delete failed (${response.status})`,
      );
    }
    // Reload rather than removing the card: the summary counts and the
    // download-all button both depend on what is left.
    load();
  } catch (error) {
    button.disabled = false;
    button.textContent = "Delete";
    list.prepend(el("p", "error-box", error.message || "Delete failed."));
  }
}

function card(doc) {
  const item = el("article", "card");

  const head = el("div", "card-head");
  head.append(el("h2", "card-title", doc.title || "Untitled"));

  const actions = el("div", "card-buttons");
  const link = el("a", "button small", "Download JSON");
  link.href = `/documents/${doc.document_id}/embeddings`;
  actions.append(link);

  const remove = el("button", "button small danger", "Delete");
  remove.type = "button";
  remove.addEventListener("click", () => removeDocument(doc, remove));
  actions.append(remove);

  head.append(actions);
  item.append(head);

  const meta = el("p", "card-meta");
  meta.append(el("span", null, doc.source_uri || "no source"));
  meta.append(el("span", "dot", "·"));
  meta.append(el("span", null, `${doc.chunks.toLocaleString()} chunks`));
  meta.append(el("span", "dot", "·"));
  meta.append(el("span", null, formatDate(doc.created_at)));
  item.append(meta);

  item.append(modelChips(doc.embedded));
  return item;
}

async function load() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    const documents = await response.json();

    list.replaceChildren();

    // Set on every load, not only when there are documents: deleting the last
    // one re-runs this with an empty list, and the button would otherwise stay
    // behind offering a zip with nothing in it.
    downloadAll.hidden = !documents.length;

    if (!documents.length) {
      summary.textContent = "No documents yet.";
      const empty = el("p", "empty", "Nothing ingested. Upload a PDF to get started.");
      list.append(empty);
      return;
    }

    const chunks = documents.reduce((total, doc) => total + doc.chunks, 0);
    summary.textContent =
      `${documents.length} document${documents.length === 1 ? "" : "s"} · ` +
      `${chunks.toLocaleString()} chunks`;

    for (const doc of documents) list.append(card(doc));
  } catch (error) {
    summary.textContent = "";
    downloadAll.hidden = true;
    list.replaceChildren(el("p", "error-box", error.message || "Could not load documents."));
  }
}

load();
