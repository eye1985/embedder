const form = document.getElementById("form");
const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const chosen = document.getElementById("chosen");
const submit = document.getElementById("submit");
const modelOptions = document.getElementById("models");
const searchChoice = document.getElementById("search-choice");
const searchOptions = document.getElementById("search-models");
const statusBox = document.getElementById("status");

// An upload outlives the page that started it, so the server is asked how it
// is going rather than the page tracking it locally -- a reload would lose
// anything held here.
const POLL_MS = 1500;
let pollTimer = null;
let busy = false;
let modelBoxes = [];
// Rebuilt whenever the checked set changes, so it can only ever offer a model
// that is actually being embedded.
let searchButtons = [];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function selected() {
  return fileInput.files[0] || null;
}

function chosenModels() {
  return modelBoxes.filter(({ box }) => box.checked).map(({ box }) => box.value);
}

function searchModel() {
  const picked = searchButtons.find((button) => button.checked);
  return picked ? picked.value : null;
}

// One model embedded means no choice to make, so the radios only appear from
// two up. The current pick survives a rebuild if it is still being embedded.
function refreshSearchChoice() {
  const tables = chosenModels();
  const keep = searchModel();

  searchChoice.hidden = tables.length < 2;
  searchOptions.replaceChildren();
  searchButtons = [];

  if (searchChoice.hidden) {
    refreshSubmit();
    return;
  }

  for (const { box, label } of modelBoxes) {
    if (!box.checked) continue;

    const option = el("label", "radio");
    const button = document.createElement("input");
    button.type = "radio";
    button.name = "search-model";
    button.value = box.value;
    button.checked = box.value === keep;
    button.disabled = busy;
    button.addEventListener("change", refreshSubmit);
    option.append(button);
    option.append(el("span", "radio-name", label));
    searchOptions.append(option);
    searchButtons.push(button);
  }

  refreshSubmit();
}

// A file with no model picked would be stored and chunked but never embedded,
// so both are required -- as is the upload slot being free. With several
// models the search choice is required too: the server refuses without it
// rather than picking one invisibly.
function refreshSubmit() {
  const tables = chosenModels();
  submit.disabled =
    busy ||
    !selected() ||
    !tables.length ||
    (tables.length > 1 && !searchModel());
}

function setBusy(on) {
  busy = on;
  fileInput.disabled = on;
  drop.classList.toggle("busy", on);
  for (const { box, available } of modelBoxes) box.disabled = on || !available;
  for (const button of searchButtons) button.disabled = on;
  submit.textContent = on ? "Embedding…" : "Upload and embed";
  refreshSubmit();
}

function showSelection() {
  const file = selected();
  chosen.hidden = !file;
  if (file) chosen.textContent = `${file.name} · ${formatSize(file.size)}`;
  refreshSubmit();
}

fileInput.addEventListener("change", showSelection);

// Drag and drop writes into the same input, so there is one source of truth
// for what is about to be sent.
for (const name of ["dragenter", "dragover"]) {
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    if (!busy) drop.classList.add("over");
  });
}
for (const name of ["dragleave", "drop"]) {
  drop.addEventListener(name, () => drop.classList.remove("over"));
}
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  if (busy) return;
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    showSelection();
  }
});

// Models are fetched rather than hardcoded: which ones exist is a database
// question, and a row without a vector table cannot be embedded into.
async function loadModels() {
  try {
    const response = await fetch("/models");
    if (!response.ok) throw new Error(`Request failed (${response.status})`);
    const models = await response.json();

    modelOptions.replaceChildren();
    modelBoxes = [];

    for (const model of models) {
      const option = el("label", model.available ? "model" : "model unavailable");

      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = model.table_name;
      box.checked = model.is_upload_default;
      box.disabled = !model.available;
      box.addEventListener("change", refreshSearchChoice);
      option.append(box);

      const text = el("span", "model-text");
      text.append(el("span", "model-name", `${model.provider} · ${model.model_name}`));
      text.append(
        el(
          "span",
          "model-meta",
          model.available
            ? `${model.table_name} · ${model.dimensions}d${model.is_upload_default ? " · default" : ""}`
            : `${model.table_name} · no vector table — run python -m db.setup`,
        ),
      );
      option.append(text);

      modelOptions.append(option);
      modelBoxes.push({
        box,
        available: model.available,
        label: `${model.provider} · ${model.model_name}`,
      });
    }
    refreshSearchChoice();
  } catch (error) {
    modelOptions.replaceChildren(
      el("p", "error-box", error.message || "Could not load models."),
    );
  }
}

function showProgress(filename, elapsed) {
  const card = el("article", "card progress");
  card.append(el("h2", "card-title", `Embedding ${filename}…`));

  const meta = el("p", "card-meta");
  meta.append(
    el("span", null, elapsed ? `${elapsed.toFixed(0)}s elapsed` : "starting"),
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
  statusBox.replaceChildren(card);
}

function showError(message) {
  statusBox.replaceChildren(el("p", "error-box", message));
}

function report(result) {
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
  meta.append(el("span", null, `${result.chunks.toLocaleString()} chunks`));
  card.append(meta);

  // `embedded` counts vectors written by THIS call, so a zero means the model
  // already held every chunk -- not that anything failed. The marked chip is
  // the model the document is now searched with.
  const chips = el("div", "chips");
  for (const [table, count] of Object.entries(result.embedded)) {
    const classes = [count ? "chip" : "chip zero"];
    if (table === result.search_model) classes.push("active");
    const chip = el("span", classes.join(" "));
    chip.append(el("span", "chip-name", table));
    chip.append(el("span", "chip-count", `+${count.toLocaleString()}`));
    chips.append(chip);
  }
  card.append(chips);

  const actions = el("div", "card-actions");
  const link = el("a", "button small", "Download embeddings");
  link.href = `/documents/${result.document_id}/embeddings`;
  actions.append(link);
  const all = el("a", "link", "See all documents");
  all.href = "/static/documents.html";
  actions.append(all);
  card.append(actions);

  statusBox.replaceChildren(card);
}

function startPolling() {
  if (pollTimer === null) pollTimer = setInterval(refreshStatus, POLL_MS);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshStatus() {
  let state;
  try {
    const response = await fetch("/upload/status");
    if (!response.ok) return;
    state = await response.json();
  } catch {
    // A blip in polling should not tear down the progress view.
    return;
  }

  if (state.active) {
    setBusy(true);
    showProgress(state.filename, state.elapsed_seconds);
    startPolling();
    return;
  }

  const wasWatching = busy;
  stopPolling();
  setBusy(false);

  const last = state.last;
  if (!last) {
    if (wasWatching) statusBox.replaceChildren();
    return;
  }
  // Render the outcome when this page was watching the upload, and also on a
  // cold load that lands just after one ended -- the case where a reload
  // happens a second too late to see it running.
  if (wasWatching || last.finished_seconds_ago <= 15) {
    if (last.result) report(last.result);
    else if (last.error) showError(last.error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = selected();
  if (!file || busy) return;

  const body = new FormData();
  body.append("file", file);
  for (const table of chosenModels()) body.append("models", table);
  const searched = searchModel();
  if (searched) body.append("search_model", searched);

  setBusy(true);
  showProgress(file.name, 0);
  startPolling();

  try {
    const response = await fetch("/upload", { method: "POST", body });
    const payload = await response.json().catch(() => null);

    stopPolling();
    setBusy(false);

    if (!response.ok) {
      // FastAPI puts the reason in `detail`; validation errors make it a list.
      const detail = payload && payload.detail;
      const message = Array.isArray(detail)
        ? detail.map((d) => d.msg).join(", ")
        : detail;
      throw new Error(message || `Upload failed (${response.status})`);
    }

    report(payload);
    fileInput.value = "";
    showSelection();
  } catch (error) {
    stopPolling();
    setBusy(false);
    showError(error.message || "Upload failed.");
  }
});

// Models first: the busy state disables their checkboxes, so they have to
// exist before a running upload is reported.
(async () => {
  await loadModels();
  await refreshStatus();
})();
