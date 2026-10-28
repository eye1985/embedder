import { Marked } from "marked";

// Answers are written by the model, so the HTML it names is text, not markup:
// the block tokenizer stands down -- a stray `<div>` is then just a paragraph
// -- and anything inline comes back escaped.
//
// The quotes are not optional. Every call below but one lands inside an
// attribute, where a `"` ends the value and starts another attribute: an alt
// text of `a" onerror="..."` is an event handler if this misses it.
function escape(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Same reasoning for URLs: a `javascript:` href is a link the model wrote.
// Only the schemes a citation could plausibly use survive; the rest keep their
// text and lose the link.
function safeHref(href: string): string | null {
  // Read the way a browser reads it, not the way it looks: a URL's tabs and
  // newlines are dropped before the scheme is resolved, so `java<TAB>script:`
  // is `javascript:` and has to be tested as such. The stripped form is what
  // gets emitted, so what was checked is what the href says.
  const url = href.replace(/[\u0000-\u0020\u007f-\u009f]/g, "");
  // Anything up to the first `:` is a scheme, however little it looks like
  // one -- the alternative is a pattern that misses it and falls through to
  // the branch that allows relative links.
  const scheme = /^([^/?#]*):/.exec(url);
  if (scheme === null) return url;
  return /^(https?|mailto)$/i.test(scheme[1]) ? url : null;
}

const md = new Marked({
  gfm: true,
  // The model writes single newlines as line breaks, not as a continued
  // paragraph, which is how they read in the transcript.
  breaks: true,
  tokenizer: {
    html: () => undefined,
  },
  renderer: {
    html: (token) => escape(token.raw),

    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens);
      const url = safeHref(href);
      if (url === null) return text;
      const name = title ? ` title="${escape(title)}"` : "";
      // Opened away from the page: the transcript is the one thing a reader
      // would not want replaced by a link they followed from it.
      return `<a href="${escape(url)}"${name} target="_blank" rel="noreferrer noopener">${text}</a>`;
    },

    image({ href, title, text }) {
      const url = safeHref(href);
      if (url === null) return escape(text);
      const name = title ? ` title="${escape(title)}"` : "";
      return `<img src="${escape(url)}" alt="${escape(text)}"${name}>`;
    },
  },
});

export function renderMarkdown(source: string): string {
  return md.parse(source, { async: false });
}
