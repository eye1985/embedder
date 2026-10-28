import { askChat, type Source } from "../lib/endpoints";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { transcript } from "../components/turn/turn";
import { composer } from "../components/composer/composer";
import { readEvents } from "../lib/sse";

mountThemeToggle();

const messages = transcript(must("messages"));
const input = composer(mustBe("form", HTMLFormElement), ask);

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

  try {
    // The answer arrives as `token` events, then one `sources` naming the
    // documents it cited, then `done`.
    for await (const { event, data } of readEvents(await askChat(prompt))) {
      if (event === "token") {
        answer += (data as { t: string }).t;
        reply.write(withoutMarkers(answer));
        messages.scrollToBottom();
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
