import { askChat } from "../lib/endpoints";
import { must, mustBe } from "../lib/dom";
import { mountThemeToggle } from "../components/theme-toggle/theme-toggle";
import { transcript } from "../components/turn/turn";
import { composer } from "../components/composer/composer";

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

  try {
    const reader = (await askChat(prompt)).getReader();
    const decoder = new TextDecoder();

    // Read the stream chunk by chunk so text appears as the model produces it.
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      reply.body.textContent = answer;
      messages.scrollToBottom();
    }
    reply.body.textContent = answer + decoder.decode();
  } catch (error) {
    reply.remove();
    const message =
      error instanceof Error ? error.message : "Something went wrong.";
    messages.add("error", "Error", message);
  } finally {
    reply.body.classList.remove("pending");
    input.unlock();
  }
}