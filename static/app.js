const form = document.getElementById("form");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const messages = document.getElementById("messages");

function addMessage(role, text = "") {
  const empty = messages.querySelector(".empty");
  if (empty) empty.remove();

  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messages.appendChild(el);
  scrollToBottom();
  return el;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

input.addEventListener("input", autoGrow);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const prompt = input.value.trim();
  if (!prompt || sendButton.disabled) return;

  addMessage("user", prompt);
  input.value = "";
  autoGrow();
  sendButton.disabled = true;

  const reply = addMessage("assistant");
  reply.classList.add("pending");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    // Read the stream chunk by chunk so text appears as the model produces it.
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      reply.textContent += decoder.decode(value, { stream: true });
      scrollToBottom();
    }
    reply.textContent += decoder.decode();
  } catch (error) {
    reply.remove();
    addMessage("error", error.message || "Something went wrong.");
  } finally {
    reply.classList.remove("pending");
    sendButton.disabled = false;
    input.focus();
  }
});
