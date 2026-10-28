import { readJobStatus, uploadPdf } from "../lib/endpoints";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { dropzone } from "../components/dropzone/dropzone";
import { modelPicker } from "../components/model-picker/model-picker";
import { uploadStatus } from "../components/upload-status/upload-status";

mountThemeToggle();

// A job outlives the page that started it, so the server is asked how it is
// going rather than the page tracking it locally -- a reload would lose
// anything held here. The slot is shared with re-embeds, so what comes back
// is not necessarily this page's work.
const POLL_MS = 1500;

const form = mustBe("form", HTMLFormElement);
const submit = mustBe("submit", HTMLButtonElement);
const status = uploadStatus(must("status"));

let pollTimer: ReturnType<typeof setInterval> | null = null;
let busy = false;

const picker = modelPicker({
  options: must("models"),
  section: must("search-choice"),
  choices: must("search-models"),
  onChange: refreshSubmit,
});

const drop = dropzone({
  zone: must("drop"),
  field: mustBe("file", HTMLInputElement),
  chosen: must("chosen"),
  onChange: refreshSubmit,
});

// A file with no model picked would be stored and chunked but never embedded,
// so both are required -- as is the upload slot being free. With several
// models the search choice is required too: the server refuses without it
// rather than picking one invisibly.
function refreshSubmit(): void {
  const tables = picker.chosenModels();
  submit.disabled =
    busy ||
    !drop.file() ||
    !tables.length ||
    (tables.length > 1 && !picker.searchModel());
}

function setBusy(on: boolean): void {
  busy = on;
  drop.setBusy(on);
  picker.setBusy(on);
  submit.textContent = on ? "Embedding…" : "Upload and embed";
  refreshSubmit();
}

function startPolling(): void {
  if (pollTimer === null) pollTimer = setInterval(refreshStatus, POLL_MS);
}

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function refreshStatus(): void {
  void (async () => {
    let state;
    try {
      state = await readJobStatus();
    } catch {
      // A blip in polling should not tear down the progress view.
      return;
    }

    if (state.active) {
      // Busy either way: the slot is one per user, so a re-embed started from
      // the jobs page blocks an upload exactly as another upload would.
      setBusy(true);
      status.progress(
        state.subject ?? "your file",
        state.elapsed_seconds,
        state.kind ?? "upload",
      );
      startPolling();
      return;
    }

    const wasWatching = busy;
    stopPolling();
    setBusy(false);

    const last = state.last;
    if (!last) {
      if (wasWatching) status.clear();
      return;
    }
    // Render the outcome when this page was watching the job, and also on a
    // cold load that lands just after one ended -- the case where a reload
    // happens a second too late to see it running.
    if (wasWatching || last.finished_seconds_ago <= 15) {
      // Only an upload's outcome belongs under the upload form -- a re-embed
      // or a restored export reports on the jobs page, and either can hold
      // the slot this page is watching. `reused_existing_document` is what an
      // upload's result has and no other job's does.
      if (
        last.kind === "upload" &&
        last.result &&
        "reused_existing_document" in last.result
      ) {
        status.report(last.result);
      } else if (last.error && last.kind === "upload") {
        status.error(last.error);
      } else {
        status.clear();
      }
    }
  })();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const picked = drop.file();
  if (!picked || busy) return;

  const body = new FormData();
  body.append("file", picked);
  for (const table of picker.chosenModels()) body.append("models", table);
  const searched = picker.searchModel();
  if (searched) body.append("search_model", searched);

  setBusy(true);
  status.progress(picked.name, 0, "upload");
  startPolling();

  void (async () => {
    try {
      const result = await uploadPdf(body);
      stopPolling();
      setBusy(false);
      status.report(result);
      drop.clear();
    } catch (error) {
      stopPolling();
      setBusy(false);
      status.error(error instanceof Error ? error.message : "Upload failed.");
    }
  })();
});

// Models first: the busy state disables their checkboxes, so they have to
// exist before a running upload is reported.
await picker.load();
refreshStatus();
