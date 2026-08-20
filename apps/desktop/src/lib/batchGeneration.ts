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

/** The exact durable checkpoint that the backend will run for a chapter. */
export type BatchGenerationNextStage =
  | "voice_synthesize"
  | "voice_assemble"
  | "transcript"
  | "audio_plan"
  | "stable_audio"
  | "mix"
  | "complete";

/** Ownership state for `nextStage`; terminal rows use `complete`. */
export type BatchGenerationStageState = "ready" | "running" | "complete";

/** Safe model snapshot frozen by the server when a batch is submitted. */
export interface BatchGenerationModelSettings {
  llmModelId: string;
  ttsBackend: "mimo" | "voxcpm2";
  ttsModelId: string;
}

export type BatchTtsBackend = BatchGenerationModelSettings["ttsBackend"];

export interface BatchGenerationChapter {
  chapterId: string;
  title: string;
  position: number;
  status: BatchGenerationChapterStatus;
  currentStage?: string | null;
  nextStage?: BatchGenerationNextStage | null;
  stageState?: BatchGenerationStageState | null;
  error?: string | null;
  voiceAudioPath?: string | null;
  mixedAudioPath?: string | null;
  audioAssets?: Array<Record<string, unknown>>;
  startedAt?: number | null;
  completedAt?: number | null;
  durationSeconds?: number | null;
  stageTimings?: Record<string, number>;
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
  /** Remaining shared MiMo cooldown, when a recent 429 has rate-limited it. */
  mimoCooldownSeconds?: number | null;
  /** Immutable model selection captured when this batch was submitted. */
  modelSettings?: BatchGenerationModelSettings;
  chapters: BatchGenerationChapter[];
  reused?: boolean;
}

const DISPLAY_STAGE_BY_NEXT_STAGE: Record<BatchGenerationNextStage, string | null> = {
  voice_synthesize: "voice",
  voice_assemble: "voice",
  transcript: "transcript",
  audio_plan: "audio_plan",
  stable_audio: "stable_audio",
  mix: "mix",
  complete: null,
};

/**
 * Normalize a durable checkpoint to the existing user-facing stage names.
 * Keeping this mapping beside the wire type prevents each UI consumer from
 * inferring backend checkpoint names independently.
 */
export function batchChapterDisplayStage(
  chapter: Pick<BatchGenerationChapter, "currentStage" | "nextStage">,
): string | null {
  if (chapter.currentStage) return chapter.currentStage;
  return chapter.nextStage ? DISPLAY_STAGE_BY_NEXT_STAGE[chapter.nextStage] : null;
}

/** Legacy rows have no snapshot and therefore retain their MiMo behavior. */
export function batchTtsBackend(
  batch: Pick<BatchGenerationResponse, "modelSettings">,
): BatchTtsBackend {
  return batch.modelSettings?.ttsBackend === "voxcpm2" ? "voxcpm2" : "mimo";
}

export function batchTtsBackendLabel(backend: BatchTtsBackend): string {
  return backend === "voxcpm2" ? "VoxCPM2" : "MiMo Voice Clone";
}

export function isBatchChapterWaitingForTts(
  chapter: Pick<BatchGenerationChapter, "nextStage" | "stageState" | "status">,
): boolean {
  return chapter.status !== "succeeded"
    && chapter.status !== "failed"
    && chapter.status !== "cancelled"
    && chapter.nextStage === "voice_synthesize"
    && chapter.stageState === "ready";
}

export function isBatchChapterTtsInProgress(
  chapter: Pick<BatchGenerationChapter, "nextStage" | "stageState">,
): boolean {
  return chapter.nextStage === "voice_synthesize" && chapter.stageState === "running";
}

export function isBatchChapterWaitingForMiMo(
  chapter: Pick<BatchGenerationChapter, "nextStage" | "stageState" | "status">,
  backend: BatchTtsBackend = "mimo",
): boolean {
  return backend === "mimo" && isBatchChapterWaitingForTts(chapter);
}

export function isBatchChapterMiMoInProgress(
  chapter: Pick<BatchGenerationChapter, "nextStage" | "stageState">,
  backend: BatchTtsBackend = "mimo",
): boolean {
  return backend === "mimo" && isBatchChapterTtsInProgress(chapter);
}

export function isBatchChapterStageInProgress(
  chapter: Pick<BatchGenerationChapter, "status" | "currentStage" | "stageState">,
): boolean {
  // `currentStage` preserves compatibility with batches created before the
  // stage-checkpoint migration, which did not return `stageState`.
  return chapter.stageState === "running"
    || (chapter.stageState == null && chapter.status === "running" && Boolean(chapter.currentStage));
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
