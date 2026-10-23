import {
  deleteDocument,
  listDocuments,
  setSearchModel,
  setSearchable,
} from "../lib/endpoints";
import type { DocumentOut } from "../lib/endpoints";
import { count } from "../lib/format";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { bindDownload } from "../components/download/download";
import { docCard } from "../components/doc-card/doc-card";
import type { DocActions } from "../components/doc-card/doc-card";
import { emptyState } from "../components/empty-state/empty-state";
import { errorBox } from "../components/error-box/error-box";

mountThemeToggle();

const list = must("list");
const summary = must("summary");
const downloadAll = mustBe("download-all", HTMLAnchorElement);

function showError(message: string): void {
  // Replaces the previous one rather than stacking: nothing reloads the list
  // after a failed download, so repeated clicks would pile boxes up.
  list.querySelector(".error-box")?.remove();
  list.prepend(errorBox(message));
}

async function save(
  action: () => Promise<unknown>,
  fallback = "Could not save.",
): Promise<boolean> {
  try {
    await action();
    return true;
  } catch (error) {
    showError(error instanceof Error ? error.message : fallback);
    return false;
  }
}

// Both switches reload the list whether or not the write succeeded: on success
// to pick up the new state, on failure to put the control back on whatever the
// server still has. Delete is the exception -- a failed delete leaves the card
// on screen, so the card restores its own button.
function actionsFor(doc: DocumentOut): DocActions {
  const id = doc.document_id;
  return {
    onSearchable(searchable) {
      void save(() => setSearchable(id, searchable)).then(() => load());
    },
    onSearchModel(table) {
      void save(() => setSearchModel(id, table)).then(() => load());
    },
    async onDelete() {
      const ok = await save(() => deleteDocument(id), "Delete failed.");
      // Reloaded rather than removing the card: the summary counts and the
      // download-all button both depend on what is left.
      if (ok) void load();
      return ok;
    },
    onError: showError,
  };
}

// The archive link is markup, not built here, so its behaviour is attached
// once rather than per load.
bindDownload(downloadAll, showError);

async function load(): Promise<void> {
  try {
    const documents = await listDocuments();

    list.replaceChildren();

    // Set on every load, not only when there are documents: deleting the last
    // one re-runs this with an empty list, and the button would otherwise stay
    // behind offering a zip with nothing in it.
    downloadAll.hidden = !documents.length;

    if (!documents.length) {
      summary.textContent = "No documents yet.";
      list.append(
        emptyState(
          "Nothing ingested yet",
          "Upload a PDF and it will be chunked, embedded and listed here.",
        ),
      );
      return;
    }

    const chunks = documents.reduce((total, doc) => total + doc.chunks, 0);
    summary.textContent =
      `${documents.length} document${documents.length === 1 ? "" : "s"} · ` +
      `${count(chunks)} chunks`;

    for (const doc of documents) list.append(docCard(doc, actionsFor(doc)));
  } catch (error) {
    summary.textContent = "";
    downloadAll.hidden = true;
    const message =
      error instanceof Error ? error.message : "Could not load documents.";
    list.replaceChildren(errorBox(message));
  }
}

void load();
