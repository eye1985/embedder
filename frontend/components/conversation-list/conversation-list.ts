import { el } from "../../lib/dom";
import { count, formatDate } from "../../lib/format";
import type { ConversationOut } from "../../lib/endpoints";

export interface ConversationActions {
  onOpen: (id: string) => void;
  onNew: () => void;
  /** Resolves false when the rename failed, so the old name can come back. */
  onRename: (id: string, title: string) => Promise<boolean>;
  onDelete: (id: string) => Promise<boolean>;
}

export interface ConversationList {
  /** Replaces the list. `active` is the conversation on screen, if any. */
  render: (items: ConversationOut[], active: string | null) => void;
}

// Icons rather than words. Two labelled buttons on a row this narrow take more
// width than the title they sit beside, and the title is what the row is for.
// Drawn in the same 20-unit stroked style as the theme toggle, so they read as
// part of the same set.
const PENCIL = `
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M13.4 2.9a1.6 1.6 0 0 1 2.3 0l1.4 1.4a1.6 1.6 0 0 1 0 2.3l-8.4 8.4-4 1 1-4 8.4-8.4Z" />
    <path d="M12.3 4l3.7 3.7" />
  </svg>`;

const TRASH = `
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M3.6 5.9h12.8" />
    <path d="M8.1 3.4h3.8" />
    <path d="M5.5 5.9l.7 10a1 1 0 0 0 1 .9h5.6a1 1 0 0 0 1-.9l.7-10" />
    <path d="M8.4 8.8v5M11.6 8.8v5" />
  </svg>`;

const PLUS = `
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"
       stroke-linecap="round" aria-hidden="true">
    <path d="M10 4.6v10.8M4.6 10h10.8" />
  </svg>`;

// A conversation with no name yet. One exists only between starting a chat and
// the first answer being stored, so it is brief -- but it is on screen for the
// whole of the first question, which is exactly when a blank row looks broken.
const UNTITLED = "New conversation";

function title(item: ConversationOut): string {
  return item.title ?? UNTITLED;
}

// When a conversation was last talked in, as a heading over the run of rows
// that share it. A timestamp on every row says the same thing seven times in
// grey; a heading every few rows says it once and leaves the rows to their
// titles. The list already arrives ordered by this, so the runs are contiguous
// and grouping is a scan rather than a sort.
const BUCKETS: [string, number][] = [
  ["Today", 0],
  ["Yesterday", 1],
  ["Previous 7 days", 7],
  ["Previous 30 days", 30],
];

function bucket(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.valueOf())) return "Older";

  // Compared by calendar day, not by elapsed hours: "Yesterday" means the
  // previous date, and at 00:30 the thing said at 23:00 was yesterday.
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);
  const days = Math.floor((midnight.valueOf() - date.valueOf()) / 86_400_000) + 1;

  for (const [label, within] of BUCKETS) {
    if (days <= within) return label;
  }
  return "Older";
}

// The title is clamped to two lines and the time is abbreviated, so the whole
// of both goes here. The message count lives only here: it tells a one-off
// question from a long thread, which is worth having but not worth a line of
// its own on every row.
function tooltip(item: ConversationOut): string {
  const turns = item.messages === 1 ? "1 message" : `${count(item.messages)} messages`;
  return `${title(item)}\n${turns} · ${formatDate(item.updated_at)}`;
}

function iconButton(markup: string, label: string, extra = ""): HTMLButtonElement {
  const button = Object.assign(el("button", `conversation-action ${extra}`.trim()), {
    type: "button",
    title: label,
  });
  button.setAttribute("aria-label", label);
  button.innerHTML = markup;
  return button;
}

// The rename control, in place of the row's label. Committed on Enter or on
// losing focus, abandoned on Escape -- a rename is one field, so a form with
// buttons around it would be more chrome than the task.
function renameField(
  current: string,
  commit: (title: string) => void,
  cancel: () => void,
): HTMLInputElement {
  const field = Object.assign(el("input", "conversation-rename"), {
    type: "text",
    value: current,
    ariaLabel: "Conversation name",
  });

  let done = false;
  function finish(save: boolean) {
    if (done) return;
    done = true;
    const next = field.value.trim();
    if (save && next && next !== current) commit(next);
    else cancel();
  }

  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      finish(false);
    }
  });
  field.addEventListener("blur", () => finish(true));
  return field;
}

export function conversationList(
  node: HTMLElement,
  actions: ConversationActions,
): ConversationList {
  const items = el("div", "conversation-items");

  const start = Object.assign(el("button", "button conversation-new"), {
    type: "button",
  });
  start.innerHTML = PLUS;
  start.append(el("span", null, "New chat"));
  start.addEventListener("click", () => actions.onNew());

  // No "Conversations" heading over the list: the date headings below are
  // already its labels, and two tiers of them in a column this narrow is one
  // more than the list needs.
  node.append(start, items);

  function row(item: ConversationOut, active: string | null): HTMLElement {
    const id = item.conversation_id;
    const entry = el("div", `conversation${id === active ? " current" : ""}`);

    const open = Object.assign(el("button", "conversation-open"), {
      type: "button",
      title: tooltip(item),
    });
    open.append(el("span", "conversation-title", title(item)));
    open.addEventListener("click", () => actions.onOpen(id));

    const tools = el("div", "conversation-actions");

    const rename = iconButton(PENCIL, "Rename");
    rename.addEventListener("click", () => {
      const current = title(item);
      const field = renameField(
        current,
        async (next) => {
          // Shown at once, then put back if the server refuses -- a success
          // replaces this row wholesale when the list reloads.
          open.querySelector(".conversation-title")!.textContent = next;
          field.replaceWith(open);
          if (!(await actions.onRename(id, next))) {
            open.querySelector(".conversation-title")!.textContent = current;
          }
        },
        () => field.replaceWith(open),
      );
      open.replaceWith(field);
      field.focus();
      field.select();
      // Selecting puts the caret at the end, which scrolls a long name so
      // only its tail is showing. The start is the part worth seeing.
      field.scrollLeft = 0;
    });

    const remove = iconButton(TRASH, "Delete", "danger");
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      if (!(await actions.onDelete(id))) remove.disabled = false;
    });

    tools.append(rename, remove);
    entry.append(open, tools);
    return entry;
  }

  return {
    render(list, active) {
      items.replaceChildren();
      if (list.length === 0) {
        items.append(
          el(
            "p",
            "conversation-none",
            "Nothing yet. Ask a question and it will be kept here.",
          ),
        );
        return;
      }
      let heading: string | null = null;
      for (const item of list) {
        const when = bucket(item.updated_at);
        if (when !== heading) {
          heading = when;
          items.append(el("p", "section-label conversation-group", when));
        }
        items.append(row(item, active));
      }
    },
  };
}
