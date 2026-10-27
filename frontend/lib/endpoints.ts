import { getJSON, postForm, postStream, send } from "./api";
import type { components } from "./api-types";

type Schemas = components["schemas"];

export type DocumentOut = Schemas["DocumentOut"];
export type ModelOut = Schemas["ModelOut"];
export type JobStatusOut = Schemas["JobStatus"];
export type JobOut = Schemas["JobOut"];
export type ImportResult = Schemas["ImportResult"];

// Every API URL the frontend knows, in one place and typed from the schema
// `npm run api-types` generates. A route renamed on the server fails the
// typecheck here instead of returning 404 in the browser.
export const listDocuments = () => getJSON<DocumentOut[]>("/api/documents");

export const listModels = () => getJSON<ModelOut[]>("/api/models");

// One slot for uploads and re-embeds both, so this answers for either.
export const readJobStatus = () => getJSON<JobStatusOut>("/api/jobs/current");

export const listJobs = () => getJSON<JobOut[]>("/api/jobs");

export const rerunJob = (id: number) =>
  send<Schemas["RerunResult"]>(`/api/jobs/${id}/rerun`, "POST");

export const deleteJob = (id: number) =>
  send<Schemas["DeleteJobResult"]>(`/api/jobs/${id}`, "DELETE");

export const askChat = (prompt: string) => postStream("/api/chat", { prompt });

// One document a chat answer cited. Not generated from the schema: it arrives
// inside the event stream, which FastAPI does not describe.
export interface Source {
  document_id: string;
  title: string | null;
  /** What the download arrives as. Null once the stored file is gone. */
  filename: string | null;
  /** In the PDF's own numbering, so the first page is 1. */
  pages: number[];
  has_source: boolean;
}

export const uploadPdf = (body: FormData) =>
  postForm<ImportResult>("/api/upload", body);

export const setSearchable = (id: string, searchable: boolean) =>
  send<Schemas["SearchableResult"]>(`/api/documents/${id}`, "PATCH", {
    searchable,
  });

export const setSearchModel = (id: string, table: string) =>
  send<Schemas["SearchModelResult"]>(
    `/api/documents/${id}/search-model/${table}`,
    "PUT",
  );

// Embeds chunks that are already stored, so no PDF is read again. Holds the
// same one-job-per-user slot an upload does, so this 409s while one runs.
export const embedExisting = (id: string, models: string[]) =>
  send<Schemas["EmbedResult"]>(`/api/documents/${id}/embeddings`, "POST", {
    models,
  });

export const deleteDocument = (id: string) =>
  send<Schemas["DeleteDocumentResult"]>(`/api/documents/${id}`, "DELETE");

// Hrefs, not readers: these responses are files, never parsed here. On the
// documents page they are saved through `download()` so a failure can be
// reported in the page -- see components/download. The download-all link is
// declared in documents.html, which owns that markup.
export const documentExportHref = (id: string) =>
  `/api/documents/${id}/embeddings`;
export const documentSourceHref = (id: string) => `/api/documents/${id}/source`;
export const documentsPageHref = "/documents";
export const jobsPageHref = "/jobs";
