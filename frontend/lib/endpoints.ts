import { getJSON, postForm, postStream, send } from "./api";
import type { components } from "./api-types";

type Schemas = components["schemas"];

export type DocumentOut = Schemas["DocumentOut"];
export type ModelOut = Schemas["ModelOut"];
export type JobStatusOut = Schemas["JobStatus"];
export type JobOut = Schemas["JobOut"];
export type ImportResult = Schemas["ImportResult"];
export type RestoreResult = Schemas["RestoreResult"];
export type ConversationOut = Schemas["ConversationOut"];
export type ConversationDetail = Schemas["ConversationDetail"];
export type MessageOut = Schemas["MessageOut"];

// One document a chat answer cited. The same shape whether it arrives inside
// the chat event stream or on a reopened conversation, which is why it is
// declared on the server as `ChatSource` and read from the schema here.
export type Source = Schemas["ChatSource"];

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

// A null `conversationId` starts a conversation. Either way the stream's
// `conversation` event names the one the turn landed in.
export const askChat = (prompt: string, conversationId: string | null) =>
  postStream("/api/chat", { prompt, conversation_id: conversationId });

export const listConversations = () =>
  getJSON<ConversationOut[]>("/api/conversations");

export const readConversation = (id: string) =>
  getJSON<ConversationDetail>(`/api/conversations/${id}`);

export const renameConversation = (id: string, title: string) =>
  send<Schemas["RenameResult"]>(`/api/conversations/${id}`, "PATCH", { title });

export const deleteConversation = (id: string) =>
  send<Schemas["DeleteConversationResult"]>(
    `/api/conversations/${id}`,
    "DELETE",
  );

export const uploadPdf = (body: FormData) =>
  postForm<ImportResult>("/api/upload", body);

// The reverse of `documentExportHref`: the downloaded JSON, put back. Nothing
// is embedded -- the vectors are in the file -- so this costs nothing to
// repeat. A document already stored comes back as a 409 until `overwrite` is
// set, which is how the page knows to ask.
export const importDocument = (file: File, overwrite = false) => {
  const body = new FormData();
  body.append("file", file);
  if (overwrite) body.append("overwrite", "true");
  return postForm<RestoreResult>("/api/documents/import", body);
};

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
