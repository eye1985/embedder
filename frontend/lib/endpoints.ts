import { getJSON, postForm, postStream, send } from "./api";
import type { components } from "./api-types";

type Schemas = components["schemas"];

export type DocumentOut = Schemas["DocumentOut"];
export type ModelOut = Schemas["ModelOut"];
export type UploadStatusOut = Schemas["UploadStatus"];
export type ImportResult = Schemas["ImportResult"];

// Every API URL the frontend knows, in one place and typed from the schema
// `npm run api-types` generates. A route renamed on the server fails the
// typecheck here instead of returning 404 in the browser.
export const listDocuments = () => getJSON<DocumentOut[]>("/api/documents");

export const listModels = () => getJSON<ModelOut[]>("/api/models");

export const readUploadStatus = () =>
  getJSON<UploadStatusOut>("/api/upload/status");

export const askChat = (prompt: string) => postStream("/api/chat", { prompt });

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

export const deleteDocument = (id: string) =>
  send<Schemas["DeleteDocumentResult"]>(`/api/documents/${id}`, "DELETE");

// Followed by the browser rather than fetched: this response sets
// Content-Disposition, so it is an href and never parsed here. The
// download-all link is declared in documents.html, which owns that markup.
export const documentExportHref = (id: string) =>
  `/api/documents/${id}/embeddings`;
export const documentsPageHref = "/documents";
