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
}

export interface AnalysisState {
  characters: CharacterMeta[];
  voices: VoiceMeta[];
  scriptPaths: Record<string, string>;
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
  voiceDescription?: string | null;
  confidence: number;
  aliases: string;
}

export type AppView =
  | { page: "library" }
  | { page: "bookDetail"; bookId: string };
