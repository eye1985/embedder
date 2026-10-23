import { download } from "../../lib/api";
import { el } from "../../lib/dom";

type OnError = (message: string) => void;

/** Make an `<a>` save through `download()` instead of navigating.
 *
 *  Every download in the app goes through here: a link straight to one of the
 *  file routes navigates away and renders the error JSON in place of the page,
 *  which on the chat page would take the transcript with it.
 *
 *  The href is left on the element: "save link as" and a page whose script
 *  failed to load both still reach the route, they just lose the error path.
 *  Reported through `onError` rather than shown here, so each page decides
 *  where an error goes.
 */
export function bindDownload(link: HTMLAnchorElement, onError: OnError): void {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    if (link.classList.contains("downloading")) return;

    // A class rather than a label, because what a link holds differs -- a
    // button is its own text, a source chip has markup inside it.
    link.classList.add("downloading");
    link.ariaBusy = "true";

    void download(link.href)
      .catch((error: unknown) =>
        onError(
          error instanceof Error ? error.message : "The download failed.",
        ),
      )
      .finally(() => {
        link.classList.remove("downloading");
        link.ariaBusy = null;
      });
  });
}

export function downloadLink(
  label: string,
  href: string,
  onError: OnError,
): HTMLAnchorElement {
  const link = el("a", "button small", label);
  link.href = href;
  bindDownload(link, onError);
  return link;
}
