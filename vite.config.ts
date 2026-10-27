import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = (path: string) => fileURLToPath(new URL(path, import.meta.url));

// Source lives in frontend/ and is built into static/, which FastAPI serves
// under /static. The HTML is rendered by Jinja rather than by Vite, so the
// build emits a manifest and the templates resolve hashed filenames through
// it -- see api/assets.py.
export default defineConfig({
  root: here("frontend"),
  base: "/static/",
  build: {
    outDir: here("static"),
    emptyOutDir: true,
    manifest: true,
    // Vite 8 bundles with Rolldown; oxc is its minifier and needs no extra dep.
    minify: "oxc",
    cssMinify: true,
    sourcemap: true,
    rollupOptions: {
      // One entry per page, plus the stylesheet every page links. Naming the
      // CSS as its own entry keeps it a real <link> in the markup instead of
      // something a module injects after first paint.
      input: {
        style: here("frontend/main.css"),
        chat: here("frontend/pages/chat.ts"),
        documents: here("frontend/pages/documents.ts"),
        upload: here("frontend/pages/upload.ts"),
        jobs: here("frontend/pages/jobs.ts"),
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    // The pages are served by FastAPI, so assets are requested cross-origin
    // in dev and need an absolute origin in the emitted URLs.
    origin: "http://localhost:5173",
  },
});
