const list = document.getElementById("list");
const summary = document.getElementById("summary");
const downloadAll = document.getElementById("download-all");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const DATE = new Intl.DateTimeFormat(undefined, {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? iso : DATE.format(date);
}

function showError(message) {
  list.prepend(el("p", "error-box", message));
}

function emptyState() {
  const box = el("div", "empty");
  box.append(el("p", "empty-title", "Nothing ingested yet"));
  box.append(
    el(
      "p",
      "empty-note",
      "Upload a PDF and it will be chunked, embedded and listed here.",
    ),
  );
  return box;
}

// Every switch on this page writes through the same path: on failure the
// caller puts its control back where it was, so what is on screen never
// claims a state the server did not accept.
async function send(url, method, body, onError = showError) {
  try {
    const response = await fetch(url, {
      method,
      // A PUT that says everything in its path carries no body, and sending
      // a JSON content-type with none makes FastAPI reject it as malformed.
      ...(body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(
        (payload && payload.detail) || `Request failed (${response.status})`,
      );
    }
    return true;
  } catch (error) {
    onError(error.message || "Could not save.");
    return false;
  }
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
function modelChoice(doc) {
  const row = el("div", "chips");
  const tables = Object.entries(doc.embedded).filter(([, count]) => count);

  if (!tables.length) {
    row.append(el("span", "chip zero", "no embeddings — not searchable"));
    return row;
  }

  for (const [table, count] of tables) {
    const chosen = table === doc.search_model;
    const option = el("label", chosen ? "chip active" : "chip");

    const button = document.createElement("input");
    button.type = "radio";
    button.name = `search-model-${doc.document_id}`;
    button.value = table;
    button.checked = chosen;
    button.disabled = !doc.searchable || (tables.length === 1 && chosen);
    button.addEventListener("change", () => choose(doc, table, row));
    option.append(button);

    option.append(el("span", "chip-name", table));
    option.append(el("span", "chip-count", count.toLocaleString()));
    row.append(option);
  }
  return row;
}

async function choose(doc, table, row) {
  for (const button of row.querySelectorAll("input")) button.disabled = true;

  const ok = await send(
    `/documents/${doc.document_id}/search-model/${table}`,
    "PUT",
  );
  // Reloaded either way: on success to move the mark, on failure to put the
  // radios back on whatever the server still has.
  await load();
  return ok;
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
    showError(error.message || "Delete failed.");
  }
}

// The master switch. Independent of the per-model ones below: both have to be
// on for a chunk to be retrievable, and turning this off leaves the per-model
// state alone so turning it back on restores it.
function searchSwitch(doc) {
  const label = el("label", "switch");

  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = doc.searchable;
  box.addEventListener("change", async () => {
    box.disabled = true;
    await send(`/documents/${doc.document_id}`, "PATCH", {
      searchable: box.checked,
    });
    // Reloaded either way: the model radios are only live while the document
    // is searchable, and on failure this puts the switch back on whatever the
    // server still has.
    await load();
  });

  label.append(box);
  label.append(el("span", null, "Searchable"));
  return label;
}

function card(doc) {
  const item = el("article", "doc");

  const head = el("div", "doc-head");
  head.append(el("h2", "doc-title", doc.title || "Untitled"));
  head.append(searchSwitch(doc));
  item.append(head);

  const meta = el("p", "doc-meta");
  meta.append(el("span", null, doc.source_uri || "no source"));
  meta.append(el("span", null, `${doc.chunks.toLocaleString()} chunks`));
  meta.append(el("span", null, formatDate(doc.created_at)));
  item.append(meta);

  // Chips and the row's own actions share a line, so a document reads as one
  // block rather than four stacked rows.
  const foot = el("div", "doc-foot");
  foot.append(modelChoice(doc));

  const actions = el("div", "doc-actions");
  const link = el("a", "button small", "Download JSON");
  link.href = `/documents/${doc.document_id}/embeddings`;
  actions.append(link);

  const remove = el("button", "button small danger", "Delete");
  remove.type = "button";
  remove.addEventListener("click", () => removeDocument(doc, remove));
  actions.append(remove);

  foot.append(actions);
  item.append(foot);

  if (!doc.searchable) item.classList.add("off");
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
      list.append(emptyState());
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
