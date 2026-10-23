import { askChat, type Source } from "../lib/endpoints";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { transcript } from "../components/turn/turn";
import { composer } from "../components/composer/composer";
import { readEvents } from "../lib/sse";

mountThemeToggle();

const messages = transcript(must("messages"));
const input = composer(mustBe("form", HTMLFormElement), ask);

async function ask(prompt: string): Promise<void> {
  messages.add("user", "You", prompt);

  const reply = messages.add("assistant", "Assistant");
  reply.body.classList.add("pending");

  // Accumulated here rather than read back off the node: textContent is
  // nullable, and one assignment per chunk is cheaper than a read and a write.
  let answer = "";
  let failed: string | null = null;
  let finished = false;

  try {
    // The answer arrives as `token` events, then one `sources` naming the
    // documents behind it, then `done`.
    for await (const { event, data } of readEvents(await askChat(prompt))) {
      if (event === "token") {
        answer += (data as { t: string }).t;
        reply.body.textContent = answer;
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
