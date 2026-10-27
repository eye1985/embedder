import {
  deleteDocument,
  embedExisting,
  listDocuments,
  listModels,
  readJobStatus,
  setSearchModel,
  setSearchable,
} from "../lib/endpoints";
import type { DocumentOut } from "../lib/endpoints";
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
      stopPolling();
      return;
    }

    const chunks = documents.reduce((total, doc) => total + doc.chunks, 0);
    const running = job.active
      ? ` · embedding ${job.subject ?? "something"}…`
      : "";
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

    // Only while something is in flight: with the slot free there is nothing
    // left to update.
    if (job.active) startPolling();
    else stopPolling();
  } catch (error) {
    stopPolling();
    summary.textContent = "";
    downloadAll.hidden = true;
    const message =
      error instanceof Error ? error.message : "Could not load documents.";
    list.replaceChildren(errorBox(message));
  }
}

void load();
