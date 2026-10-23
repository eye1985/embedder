export interface ServerEvent {
  event: string;
  data: unknown;
}

// Splits an event-stream body into its events. A read can land anywhere --
// mid-frame, mid-character -- so the tail is held back until its blank-line
// terminator arrives, and the decoder is told the text is still coming.
export async function* readEvents(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<ServerEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    for (
      let end = buffer.indexOf("\n\n");
      end !== -1;
      end = buffer.indexOf("\n\n")
    ) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const parsed = parse(frame);
      if (parsed !== null) yield parsed;
    }
  }
}

function parse(frame: string): ServerEvent | null {
  let event = "message";
  const data: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // One space after the colon is framing, not part of the value.
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    // A frame we cannot read is dropped rather than thrown: the stream is
    // still running, and the next event may well be fine.
    return null;
  }
}
