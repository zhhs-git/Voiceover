import { invoke } from "./platform";

export type BatchGenerationStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "completed_with_errors"
  | "cancelled"
  | "failed";

export type BatchGenerationChapterStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface BatchGenerationChapter {
  chapterId: string;
  title: string;
  position: number;
  status: BatchGenerationChapterStatus;
  currentStage?: string | null;
  error?: string | null;
  voiceAudioPath?: string | null;
  mixedAudioPath?: string | null;
  audioAssets?: Array<Record<string, unknown>>;
  startedAt?: number | null;
  completedAt?: number | null;
}

export interface BatchGenerationResponse {
  batchId?: string;
  bookId?: string;
  status: BatchGenerationStatus;
  force?: boolean;
  cacheSegments?: boolean;
  cancelRequested?: boolean;
  currentChapterId?: string | null;
  currentStage?: string | null;
  error?: string | { message?: string } | null;
  createdAt?: number;
  updatedAt?: number;
  completedAt?: number | null;
  totalCount?: number;
  completedCount?: number;
  succeededCount?: number;
  failedCount?: number;
  cancelledCount?: number;
  chapters: BatchGenerationChapter[];
  reused?: boolean;
}

export interface StartBatchGenerationInput {
  bookId: string;
  chapterIds: string[];
  force?: boolean;
  cacheSegments?: boolean;
}

export async function startBatchGeneration(
  input: StartBatchGenerationInput,
): Promise<BatchGenerationResponse> {
  return invoke<BatchGenerationResponse>("batch_generation_start", { ...input });
}

export async function getBatchGenerationStatus(
  batchId: string,
): Promise<BatchGenerationResponse> {
  return invoke<BatchGenerationResponse>("batch_generation_status", { batchId });
}

export async function getActiveBatchGeneration(
  bookId: string,
): Promise<BatchGenerationResponse | null> {
  return invoke<BatchGenerationResponse | null>("batch_generation_active", { bookId });
}

export async function cancelBatchGeneration(
  batchId: string,
): Promise<BatchGenerationResponse> {
  return invoke<BatchGenerationResponse>("batch_generation_cancel", { batchId });
}

export function isActiveBatchGeneration(
  status: BatchGenerationStatus,
): boolean {
  return status === "queued" || status === "running";
}

export function batchErrorMessage(response: BatchGenerationResponse): string {
  if (typeof response.error === "string" && response.error.trim()) return response.error;
  if (
    response.error &&
    typeof response.error !== "string" &&
    typeof response.error.message === "string"
  ) {
    return response.error.message;
  }
  return "批量生成失败。";
}
