import { el } from "../../lib/dom";
import { count, formatDate } from "../../lib/format";
import { documentsPageHref } from "../../lib/endpoints";
import type { JobOut } from "../../lib/endpoints";
import { chip, chipRow } from "../chips/chips";

export interface JobActions {
  /** Resolves false when the re-run failed, the one case where this card's
   *  button has to come back. */
  onRerun: () => Promise<boolean>;
  onDelete: () => Promise<boolean>;
}

const KIND_LABEL: Record<string, string> = {
  upload: "Upload",
  embed: "Re-embed",
};

const STATUS_LABEL: Record<string, string> = {
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  interrupted: "Interrupted",
};

function elapsed(job: JobOut): string {
  const seconds = job.elapsed_seconds;
  if (seconds < 90) return `${seconds.toFixed(0)}s`;
  return `${(seconds / 60).toFixed(0)}m`;
}

// One chip per model the job was asked to embed with. The count comes from the
// job's own result, so it is what THIS run wrote -- a zero means the model
// already held every chunk, not that anything failed. A job with no result
// yet, or one that died before writing, shows the model with no count.
function models(job: JobOut): HTMLElement {
  const row = chipRow();
  const embedded = job.result?.embedded;

  for (const table of job.models) {
    const n = embedded?.[table];
    row.append(
      chip({
        name: table,
        count: n === undefined ? "—" : `+${count(n)}`,
        zero: !n,
      }),
    );
  }
  return row;
}

function rerunWarning(job: JobOut): string {
  return (
    `Re-run this ${KIND_LABEL[job.kind]?.toLowerCase() ?? job.kind} of ` +
    `"${job.subject}"?\n\n` +
    `Chunks a model already holds are skipped, so only missing embeddings ` +
    `are paid for — but this does spend against the embedding API.`
  );
}

// A finished row is only history: deleting it touches no document, no chunk
// and no vector. A running one is not history yet -- deleting that frees the
// slot without stopping the work, so it is labelled and warned about
// separately.
function deleteLabel(job: JobOut): string {
  return job.status === "running" ? "Force release" : "Delete history";
}

function deleteWarning(job: JobOut): string {
  if (job.status !== "running") return "";
  return (
    `Force the running job "${job.subject}" out of the way?\n\n` +
    `This frees the slot so you can start another job. It does NOT stop work ` +
    `that is still going — nothing running reads this row. Only do this when ` +
    `the job is stuck because the server restarted.`
  );
}

export function jobCard(job: JobOut, actions: JobActions): HTMLElement {
  const item = el("article", `job ${job.status}`);

  const head = el("div", "job-head");
  head.append(el("h2", "job-title", job.subject));
  head.append(
    el("span", `badge ${job.status}`, STATUS_LABEL[job.status] ?? job.status),
  );
  item.append(head);

  const meta = el("p", "job-meta");
  meta.append(el("span", null, KIND_LABEL[job.kind] ?? job.kind));
  meta.append(el("span", null, formatDate(job.started_at)));
  meta.append(el("span", null, elapsed(job)));
  if (job.result) meta.append(el("span", null, `${count(job.result.chunks)} chunks`));
  item.append(meta);

  if (job.models.length) item.append(models(job));

  // Shown for an interrupted job too: the message there is the restart, which
  // is the whole reason the card offers a re-run.
  if (job.error) item.append(el("p", "job-error", job.error));

  const foot = el("div", "job-foot");
  const actionRow = el("div", "job-actions");

  if (job.can_rerun) {
    const rerun = el("button", "button small", "Re-run");
    rerun.type = "button";
    rerun.addEventListener("click", () => {
      if (!window.confirm(rerunWarning(job))) return;
      rerun.disabled = true;
      rerun.textContent = "Embedding…";
      void actions.onRerun().then((ok) => {
        if (!ok) {
          rerun.disabled = false;
          rerun.textContent = "Re-run";
        }
      });
    });
    actionRow.append(rerun);
  }

  if (job.document_id) {
    const link = el("a", "link", "See document");
    link.href = documentsPageHref;
    actionRow.append(link);
  }

  const label = deleteLabel(job);
  const remove = el("button", "button small danger", label);
  remove.type = "button";
  remove.addEventListener("click", () => {
    const warning = deleteWarning(job);
    if (warning && !window.confirm(warning)) return;
    remove.disabled = true;
    remove.textContent = "Removing…";
    void actions.onDelete().then((ok) => {
      if (!ok) {
        remove.disabled = false;
        remove.textContent = label;
      }
    });
  });
  actionRow.append(remove);

  foot.append(actionRow);
  item.append(foot);
  return item;
}
