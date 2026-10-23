// FastAPI puts the reason in `detail`; a validation error makes it a list.
async function reason(response: Response): Promise<string> {
  const payload: unknown = await response.json().catch(() => null);
  const detail =
    payload && typeof payload === "object" && "detail" in payload
      ? payload.detail
      : null;

  if (Array.isArray(detail)) {
    return detail
      .map((item: unknown) =>
        item && typeof item === "object" && "msg" in item ? String(item.msg) : "",
      )
      .filter(Boolean)
      .join(", ");
  }
  if (typeof detail === "string") return detail;
  return `Request failed (${response.status})`;
}

export async function getJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await reason(response));
  return (await response.json()) as T;
}

export async function send<T>(
  url: string,
  method: string,
  body?: unknown,
): Promise<T | null> {
  const response = await fetch(url, {
    method,
    // A request that says everything in its path carries no body, and a JSON
    // content-type with none makes FastAPI reject it as malformed.
    ...(body === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  if (!response.ok) throw new Error(await reason(response));
  return (await response.json().catch(() => null)) as T | null;
}

export async function postForm<T>(url: string, body: FormData): Promise<T> {
  const response = await fetch(url, { method: "POST", body });
  if (!response.ok) throw new Error(await reason(response));
  return (await response.json()) as T;
}

// Not JSON: the reply is streamed, so the caller reads the body itself.
export async function postStream(
  url: string,
  body: unknown,
): Promise<ReadableStream<Uint8Array>> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Request failed (${response.status})`);
  }
  return response.body;
}

// How long a blob URL is kept alive after the click that saves it. The click
// starts the save, it does not finish it, so revoking immediately can cancel
// a download that has not begun reading yet.
const REVOKE_AFTER_MS = 10_000;

// Every download route sets `attachment; filename="..."`. Read back here
// because a blob URL has no name of its own -- without this the file saves as
// the URL's UUID.
function attachmentName(response: Response): string {
  const header = response.headers.get("Content-Disposition") ?? "";
  return /filename="([^"]+)"/.exec(header)?.[1] ?? "download";
}

// A download, fetched rather than followed. A link straight to one of these
// routes navigates away from the page when it fails and renders the error JSON
// in its place; fetching keeps the page up and makes the reason reportable,
// at the cost of holding the file in memory while it saves.
export async function download(url: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await reason(response));

  const href = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = href;
  link.download = attachmentName(response);
  // In the document for the click: a detached anchor's `download` is ignored
  // by some browsers.
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(href), REVOKE_AFTER_MS);
}
