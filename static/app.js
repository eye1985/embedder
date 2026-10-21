const form = document.getElementById("form");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const messages = document.getElementById("messages");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Returns the body, which is what the stream writes into; the turn around it
// is reachable as its parent when a failed reply has to be pulled back out.
function addTurn(role, label, text = "") {
  const empty = messages.querySelector(".empty");
  if (empty) empty.remove();

  const turn = el("div", `turn ${role}`);
  turn.append(el("div", "turn-role", label));
  const body = el("div", "turn-body", text);
  turn.append(body);

  messages.appendChild(turn);
  scrollToBottom();
  return body;
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
  // The field grows to fit, so a bar is only wanted once max-height has
  // stopped it growing.
  input.style.overflowY =
    input.scrollHeight > input.clientHeight ? "auto" : "hidden";
}

input.addEventListener("input", autoGrow);
autoGrow();

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

  addTurn("user", "You", prompt);
  input.value = "";
  autoGrow();
  sendButton.disabled = true;

  const reply = addTurn("assistant", "Assistant");
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
    reply.parentElement.remove();
    addTurn("error", "Error", error.message || "Something went wrong.");
  } finally {
    reply.classList.remove("pending");
    sendButton.disabled = false;
    input.focus();
  }
});
