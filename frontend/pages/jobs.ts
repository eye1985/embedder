import { deleteJob, listJobs, rerunJob } from "../lib/endpoints";
import type { JobOut } from "../lib/endpoints";
import { must } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { jobCard } from "../components/job-card/job-card";
import { emptyState } from "../components/empty-state/empty-state";
import { errorBox } from "../components/error-box/error-box";

mountThemeToggle();

// A job outlives the page that started it, so what is running is asked of the
// server rather than tracked here.
const POLL_MS = 1500;

const list = must("list");
const summary = must("summary");

let pollTimer: ReturnType<typeof setInterval> | null = null;

function showError(message: string): void {
  list.querySelector(".error-box")?.remove();
  list.prepend(errorBox(message));
}

async function save(
  action: () => Promise<unknown>,
  fallback = "That did not work.",
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

// A re-run embeds inside its own request, so it can take as long as the work
// does. Polling is started before the wait rather than after it, so the new
// running row shows up immediately instead of only once the request returns.
function actionsFor(job: JobOut) {
  return {
    async onRerun() {
      startPolling();
      const ok = await save(() => rerunJob(job.job_id), "Re-run failed.");
      await load();
      return ok;
    },
    async onDelete() {
      const ok = await save(() => deleteJob(job.job_id), "Delete failed.");
      if (ok) await load();
      return ok;
    },
  };
}

function describe(jobs: JobOut[]): string {
  if (!jobs.length) return "No jobs yet.";
  const running = jobs.filter((job) => job.status === "running").length;
  const stuck = jobs.filter((job) => job.status === "interrupted").length;

  const parts = [`${jobs.length} job${jobs.length === 1 ? "" : "s"}`];
  if (running) parts.push(`${running} running`);
  if (stuck) parts.push(`${stuck} interrupted`);
  return parts.join(" · ");
}

async function load(): Promise<void> {
  let jobs;
  try {
    jobs = await listJobs();
  } catch (error) {
    stopPolling();
    summary.textContent = "";
    const message =
      error instanceof Error ? error.message : "Could not load jobs.";
    list.replaceChildren(errorBox(message));
    return;
  }

  summary.textContent = describe(jobs);
  list.replaceChildren();

  if (!jobs.length) {
    list.append(
      emptyState(
        "Nothing has run yet",
        "Uploads and re-embeds are listed here, with what they wrote and " +
          "what to do if one was interrupted.",
      ),
    );
    stopPolling();
    return;
  }

  for (const job of jobs) list.append(jobCard(job, actionsFor(job)));

  // Only while something is actually in flight: a page of finished jobs has
  // nothing left to update.
  if (jobs.some((job) => job.status === "running")) startPolling();
  else stopPolling();
}

void load();
