import {
  askChat,
  deleteConversation,
  listConversations,
  readConversation,
  renameConversation,
} from "../lib/endpoints";
import type { ConversationOut, Source } from "../lib/endpoints";
import { must, mustBe, need } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { transcript } from "../components/turn/turn";
import { composer } from "../components/composer/composer";
import { conversationList } from "../components/conversation-list/conversation-list";
import { readEvents } from "../lib/sse";

mountThemeToggle();

const messages = transcript(must("messages"));
const input = composer(mustBe("form", HTMLFormElement), ask);
const sidebar = conversationList(must("conversations"), {
  onOpen: (id) => void open(id),
  onNew: startNew,
  onRename: rename,
  onDelete: remove,
});

// The conversation on screen. Null is a chat that has not been stored yet --
// the first answer is what creates it, so until then there is nothing to name
// or to return to.
let current: string | null = null;
let conversations: ConversationOut[] = [];

// The sticky sidebar has to sit below the sticky header, and CSS cannot read
// an element's height. Watched rather than measured once: the header wraps its
// nav at narrow widths, so its height is not a constant.
const shell = need(document.querySelector<HTMLElement>(".app"), "page shell");
const header = need(document.querySelector<HTMLElement>(".header"), "header");

function syncHeaderHeight(): void {
  shell.style.setProperty("--header-h", `${header.offsetHeight}px`);
}

new ResizeObserver(syncHeaderHeight).observe(header);
syncHeaderHeight();

// The model marks each claim with the number of the document it came from so
// the server can tell which ones the answer actually used. That is all they
// are for, so they come back out before the reader sees them -- with the space
// in front, or removing one leaves a gap before the full stop.
//
// The whole accumulated answer is stripped on every token rather than each
// token on arrival: a marker can straddle two of them, and the trailing clause
// is what keeps a half-arrived one from flashing up as `[1`.
function withoutMarkers(answer: string): string {
  return answer.replace(/ ?\[\d+\]/g, "").replace(/ ?\[\d*$/, "");
}

function report(error: unknown): void {
  messages.add(
    "error",
    "Error",
    error instanceof Error ? error.message : "Something went wrong.",
  );
}

// Which conversation is open is in the URL, so one can be linked to and a
// reload comes back to it. Replaced rather than pushed: switching conversation
// is not somewhere to go Back to, and the transcript is on the server either
// way.
function showInUrl(id: string | null): void {
  history.replaceState(null, "", id === null ? "/" : `/?c=${id}`);
}

async function refresh(): Promise<void> {
  conversations = await listConversations();
  sidebar.render(conversations, current);
}

async function open(id: string): Promise<void> {
  let detail;
  try {
    detail = await readConversation(id);
  } catch (error) {
    report(error);
    return;
  }

  messages.reset();
  for (const message of detail.messages) {
    if (message.role === "user") {
      messages.add("user", "You", message.content);
      continue;
    }
    // Rendered exactly as a live answer is, markers stripped and sources
    // resolved -- the server re-resolves those on every read, so a document
    // deleted since this answer was written is simply no longer offered.
    const reply = messages.add("assistant", "Assistant");
    reply.write(withoutMarkers(message.content));
    reply.showSources(message.sources);
  }

  current = id;
  showInUrl(id);
  sidebar.render(conversations, current);
  messages.scrollToBottom();
}

function startNew(): void {
  current = null;
  messages.reset();
  showInUrl(null);
  sidebar.render(conversations, current);
}

async function rename(id: string, title: string): Promise<boolean> {
  try {
    await renameConversation(id, title);
    await refresh();
    return true;
  } catch (error) {
    report(error);
    return false;
  }
}

async function remove(id: string): Promise<boolean> {
  try {
    await deleteConversation(id);
  } catch (error) {
    report(error);
    return false;
  }
  // Only the list changes unless the deleted one was on screen, in which case
  // what is on screen no longer exists anywhere.
  if (id === current) startNew();
  await refresh();
  return true;
}

async function ask(prompt: string): Promise<void> {
  messages.add("user", "You", prompt);

  const reply = messages.add("assistant", "Assistant");
  reply.body.classList.add("pending");

  // Accumulated here rather than read back off the node: the node holds
  // rendered markdown, which is not the source the next token appends to.
  // The raw text is what accumulates, markers and all -- stripping it here
  // would mean re-stripping a string the next token appends to anyway.
  let answer = "";
  let failed: string | null = null;
  let finished = false;
  // Where the server says the turn landed. A new conversation's id arrives
  // this way, and so does a partial answer's after a failure -- which is why
  // it is acted on before the failure is.
  let landed: string | null = null;

  try {
    // The answer arrives as `token` events, then `conversation` naming the
    // conversation the turn was stored in, then one `sources` naming the
    // documents it cited, then `done`.
    for await (const { event, data } of readEvents(
      await askChat(prompt, current),
    )) {
      if (event === "token") {
        answer += (data as { t: string }).t;
        reply.write(withoutMarkers(answer));
        messages.scrollToBottom();
      } else if (event === "conversation") {
        landed = (data as { conversation_id: string }).conversation_id;
      } else if (event === "sources") {
        reply.showSources((data as { sources: Source[] }).sources);
        messages.scrollToBottom();
      } else if (event === "error") {
        failed = (data as { detail: string }).detail;
      } else if (event === "done") {
        finished = true;
        break;
      }
    }
  } catch (error) {
    failed = error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    reply.body.classList.remove("pending");
    input.unlock();
  }

  // The turn is stored by the request that answered it, so the sidebar is
  // reloaded rather than guessed at: a new conversation appears with the name
  // the server gave it, and an existing one moves to the top because it was
  // just talked in.
  if (landed !== null) {
    current = landed;
    showInUrl(landed);
    void refresh().catch(report);
  }

  // No `done` and no error means the stream went away mid-answer. Saying so
  // beats leaving a half-sentence that reads like the whole reply.
  if (failed === null && !finished) {
    failed = "The connection closed before the answer finished.";
  }
  if (failed === null) return;

  // Whatever arrived before the failure is kept, for the same reason.
  if (answer === "") reply.remove();
  messages.add("error", "Error", failed);
}

// The conversation named in the URL, or the one last talked in -- persistent
// history is only worth having if a reload comes back to where it was. A user
// with none lands on an empty chat, which is also what "New chat" gives.
async function boot(): Promise<void> {
  try {
    conversations = await listConversations();
  } catch (error) {
    report(error);
    return;
  }

  const wanted =
    new URLSearchParams(location.search).get("c") ??
    conversations[0]?.conversation_id ??
    null;

  if (wanted === null) {
    sidebar.render(conversations, current);
    return;
  }
  await open(wanted);
}

void boot();
