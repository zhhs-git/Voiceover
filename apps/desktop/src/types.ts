export interface ChapterMeta {
  id: string;
  title: string;
  textLength: number;
  textPath: string;
}

export interface CharacterMeta {
  id: string;
  canonicalName: string;
  aliases: string[];
  gender: string;
  ageClass?: string;
  identityStatus?: "provisional" | "confirmed" | "merged";
  voiceId: string;
  voiceSource?: "auto" | "manual";
  voiceAssignmentVersion?: number;
  voiceProfile?: string;
  fallbackVoiceId?: string;
  voiceDesign?: string;
  voiceDescription?: string;
  confidence: number;
}

export interface VoiceMeta {
  id: string;
  displayName: string;
  backend: string;
}

export interface VoiceOption {
  id: string;
  displayName: string;
  available?: boolean;
}

export interface BookState {
  title: string;
  bookId: string;
  workDir: string;
  chapters: ChapterMeta[];
  /** Stable narrator identity for every chapter in this book. */
  narratorVoiceId?: "narrator_female" | "narrator_male" | "narrator_default";
}

export interface AnalysisState {
  characters: CharacterMeta[];
  voices: VoiceMeta[];
  scriptPaths: Record<string, string>;
}

export interface AudioAsset {
  assetId: string;
  kind: "music" | "sfx";
  sceneId: string;
  path: string;
  /** Changes whenever a newly generated file replaces the same stable path. */
  refreshKey?: string;
  model?: string;
  durationSeconds?: number;
  signature?: string;
  cacheHit?: boolean;
}

export interface RightsResult {
  classification: string;
  reason: string;
  requiresAttestation: boolean;
  evidence: string[];
}

export interface ProgressDetail {
  label: string;
  value: string;
}

export type WorkflowStepStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "needs_review"
  | "skipped";

export interface WorkflowStep {
  id: string;
  label: string;
  status: WorkflowStepStatus;
  detail?: string;
  error?: string;
}

export interface ChapterWorkflowStatus {
  chapterId: string;
  kind: "analysis" | "generation";
  currentStep?: string;
  steps: WorkflowStep[];
  status: WorkflowStepStatus;
  detail?: string;
  error?: string;
  updatedAt?: number;
}

export type PipelineStage =
  | "idle"
  | "importing"
  | "analyzing"
  | "saving"
  | "generating"
  | "done"
  | "error";

export type WorkspaceStep = 1 | 2 | 3 | 4 | "done";

export interface LibraryBook {
  id: string;
  title: string;
  sourcePath: string;
  workDir: string;
  importedAt: string | null;
  narratorVoiceId?: "narrator_female" | "narrator_male" | "narrator_default";
}

export interface ChapterRow {
  id: string;
  title: string;
  status: string;
  scriptPath: string | null;
}

export interface CharacterRow {
  id: string;
  canonicalName: string;
  gender: string | null;
  ageClass?: string | null;
  identityStatus?: "provisional" | "confirmed" | "merged" | null;
  voiceId: string | null;
  voiceSource?: "auto" | "manual" | null;
  voiceAssignmentVersion?: number | null;
  voiceProfile?: string | null;
  fallbackVoiceId?: string | null;
  voiceDesign?: string | null;
  voiceDescription?: string | null;
  confidence: number;
  aliases: string;
}

export type AppView =
  | { page: "library" }
  | { page: "bookDetail"; bookId: string };
