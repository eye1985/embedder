import { formatSize } from "../../lib/format";

interface DropzoneOptions {
  zone: HTMLElement;
  field: HTMLInputElement;
  chosen: HTMLElement;
  onChange: () => void;
}

export interface Dropzone {
  file: () => File | null;
  clear: () => void;
  setBusy: (on: boolean) => void;
}

// Drag and drop writes into the same <input type="file">, so there is one
// source of truth for what is about to be sent.
export function dropzone({
  zone,
  field,
  chosen,
  onChange,
}: DropzoneOptions): Dropzone {
  let busy = false;

  function file(): File | null {
    return field.files?.[0] ?? null;
  }

  function show() {
    const picked = file();
    chosen.hidden = !picked;
    if (picked) {
      chosen.textContent = `${picked.name} · ${formatSize(picked.size)}`;
    }
    onChange();
  }

  field.addEventListener("change", show);

  for (const name of ["dragenter", "dragover"] as const) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      if (!busy) zone.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"] as const) {
    zone.addEventListener(name, () => zone.classList.remove("over"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (busy) return;
    const dropped = event.dataTransfer?.files;
    if (dropped && dropped.length) {
      field.files = dropped;
      show();
    }
  });

  return {
    file,
    clear() {
      field.value = "";
      show();
    },
    setBusy(on: boolean) {
      busy = on;
      field.disabled = on;
      zone.classList.toggle("busy", on);
    },
  };
}
