import {
  deleteDocument,
  embedExisting,
  importDocument,
  listDocuments,
  listModels,
  readJobStatus,
  setSearchModel,
  setSearchable,
} from "../lib/endpoints";
import type { DocumentOut } from "../lib/endpoints";
import { ApiError } from "../lib/api";
import { count } from "../lib/format";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { bindDownload } from "../components/download/download";
import { docCard } from "../components/doc-card/doc-card";
import type { DocActions, DocContext } from "../components/doc-card/doc-card";
import { emptyState } from "../components/empty-state/empty-state";
import { errorBox } from "../components/error-box/error-box";

mountThemeToggle();

// Embedding runs on the server and outlives this page, so what is in flight is
// asked of the server rather than tracked here.
const POLL_MS = 1500;

const list = must("list");
const summary = must("summary");
const downloadAll = mustBe("download-all", HTMLAnchorElement);
const importButton = mustBe("import", HTMLButtonElement);
const importFile = mustBe("import-file", HTMLInputElement);

let pollTimer: ReturnType<typeof setInterval> | null = null;

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

function startPolling(): void {
  if (pollTimer === null) pollTimer = setInterval(() => void load(), POLL_MS);
}

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
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
    // One model per request, so each gets its own outcome: a single call with
    // several would report nothing if the second failed after the first wrote.
    // Polling starts before the wait, so the running job shows up immediately
    // rather than only when the embedding returns.
    async onEmbed(table) {
      startPolling();
      const ok = await save(
        () => embedExisting(id, [table]),
        "Embedding failed.",
      );
      await load();
      return ok;
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

// Two reasons the button is unavailable, and they can both be true: this page
// is mid-import, or the slot is held by work started anywhere. Computed in one
// place so whichever clears last leaves the button in the right state.
let importing = false;
let slotHeld = false;

function refreshImport(): void {
  importButton.disabled = importing || slotHeld;
  importButton.textContent = importing ? "Importing…" : "Import JSON";
  importButton.title =
    slotHeld && !importing
      ? "Another job is running. Imports take the same slot."
      : "";
}

function setImporting(on: boolean): void {
  importing = on;
  refreshImport();
}

// Restoring an export writes rows and embeds nothing, so unlike an upload
// there is no job to poll -- the response is the whole outcome.
async function runImport(file: File, overwrite: boolean): Promise<void> {
  setImporting(true);
  try {
    await importDocument(file, overwrite);
    list.querySelector(".error-box")?.remove();
    await load();
  } catch (error) {
    // The endpoint's other 409 is a busy job slot, which is not an offer.
    if (
      !overwrite &&
      error instanceof ApiError &&
      error.code === "document_exists"
    ) {
      if (
        confirm(
          `${error.message}\n\nReplace the stored document with this file? ` +
            `Its chunks and every model's vectors are replaced by the ` +
            `file's. The PDF it was imported from stays.`,
        )
      ) {
        await runImport(file, true);
      }
      return;
    }
    showError(error instanceof Error ? error.message : "Import failed.");
  } finally {
    setImporting(false);
  }
}

importButton.addEventListener("click", () => importFile.click());

importFile.addEventListener("change", () => {
  const file = importFile.files?.[0];
  // Cleared first, so picking the same file again still fires `change` --
  // which is what a second attempt after an error looks like.
  importFile.value = "";
  if (file) void runImport(file, false);
});

async function load(): Promise<void> {
  try {
    // Models and the job slot come along because the cards need both: which
    // models could still embed a document, and whether anything may start.
    const [documents, models, job] = await Promise.all([
      listDocuments(),
      listModels(),
      readJobStatus(),
    ]);

    list.replaceChildren();

    // Decided once, before the empty-library branch returns: an import can be
    // in flight with nothing listed yet, and leaving the button disabled with
    // no poll to re-enable it would strand it.
    slotHeld = job.active;
    refreshImport();
    if (job.active) startPolling();
    else stopPolling();

    // Set on every load, not only when there are documents: deleting the last
    // one re-runs this with an empty list, and the button would otherwise stay
    // behind offering a zip with nothing in it.
    downloadAll.hidden = !documents.length;

    const running = job.active
      ? ` · ${job.kind === "import" ? "importing" : "embedding"} ` +
        `${job.subject ?? "something"}…`
      : "";

    if (!documents.length) {
      summary.textContent = `No documents yet${running}.`;
      list.append(
        emptyState(
          "Nothing ingested yet",
          "Upload a PDF and it will be chunked, embedded and listed here — " +
            "or import a JSON export to put one back.",
        ),
      );
      return;
    }

    const chunks = documents.reduce((total, doc) => total + doc.chunks, 0);
    summary.textContent =
      `${documents.length} document${documents.length === 1 ? "" : "s"} · ` +
      `${count(chunks)} chunks${running}`;

    const context: DocContext = {
      // Only models with a vector table: `db/setup.py` creates one per active
      // model, so an inactive row has nothing to embed into and the server
      // would refuse it.
      available: models.filter((model) => model.available),
      busy: job.active,
    };

    for (const doc of documents) {
      list.append(docCard(doc, actionsFor(doc), context));
    }
  } catch (error) {
    stopPolling();
    summary.textContent = "";
    downloadAll.hidden = true;
    // The slot state is unknown now, so the button is left usable rather than
    // stuck -- a job that is in fact running turns the import away with a 409.
    slotHeld = false;
    refreshImport();
    const message =
      error instanceof Error ? error.message : "Could not load documents.";
    list.replaceChildren(errorBox(message));
  }
}

void load();
