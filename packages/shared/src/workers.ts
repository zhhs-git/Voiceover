import { z } from "zod";

const BaseWorkerRequestSchema = z.object({
  bookId: z.string().min(1)
});

export const ExtractBookTextRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("extract_book_text"),
  inputPath: z.string().min(1),
  outputDirectory: z.string().min(1)
});

export const DetectChaptersRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("detect_chapters"),
  extractedTextPath: z.string().min(1),
  outputDirectory: z.string().min(1)
});

export const AnalyzeChapterRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("analyze_chapter"),
  chapterId: z.string().min(1),
  chapterTextPath: z.string().min(1),
  characterBiblePath: z.string().min(1).optional(),
  outputDirectory: z.string().min(1),
  resumeFromStage: z.enum([
    "characters",
    "voice_design",
    "speakers",
    "delivery",
    "voice_direction",
    "script",
  ]).optional()
});

export const SynthesizeSegmentAudioRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("synthesize_segment_audio"),
  chapterId: z.string().min(1),
  segmentId: z.string().min(1),
  scriptPath: z.string().min(1),
  outputDirectory: z.string().min(1),
  backend: z.string().min(1).optional(),
  modelId: z.string().min(1).optional(),
  narratorVoiceId: z.string().min(1).optional(),
  voiceProfileDirectory: z.string().min(1).optional()
});

export const TranscribeChapterAudioRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("transcribe_chapter_audio"),
  chapterId: z.string().min(1),
  scriptPath: z.string().min(1),
  voiceAudioPath: z.string().min(1),
  analysisDirectory: z.string().min(1).optional(),
  whisperModel: z.string().min(1).optional(),
  whisperPython: z.string().min(1).optional()
});

export const PlanChapterAudioRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("plan_chapter_audio"),
  chapterId: z.string().min(1),
  scriptPath: z.string().min(1),
  transcriptPath: z.string().min(1),
  chapterTextPath: z.string().min(1).optional(),
  analysisDirectory: z.string().min(1).optional()
});

export const GenerateAudioAssetsRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("generate_audio_assets"),
  chapterId: z.string().min(1),
  scriptPath: z.string().min(1),
  outputDirectory: z.string().min(1),
  mixedOutputPath: z.string().min(1).optional(),
  force: z.boolean().optional(),
  assetId: z.string().min(1).optional(),
  assetKind: z.enum(["music", "sfx"]).optional()
});

export const AssembleChapterAudioRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("assemble_chapter_audio"),
  chapterId: z.string().min(1),
  segmentAudioDirectory: z.string().min(1),
  outputPath: z.string().min(1)
});

export const MixChapterAudioRequestSchema = BaseWorkerRequestSchema.extend({
  command: z.literal("mix_chapter_audio"),
  chapterId: z.string().min(1),
  scriptPath: z.string().min(1),
  segmentAudioDirectory: z.string().min(1),
  voiceAudioPath: z.string().min(1),
  audioAssetsDirectory: z.string().min(1),
  outputPath: z.string().min(1),
  mergeSegments: z.boolean().optional(),
  maxMergedSegmentWords: z.number().int().positive().optional(),
  maxMergedSegmentCharacters: z.number().int().positive().optional(),
  gapSeconds: z.number().nonnegative().optional(),
  musicGain: z.number().nonnegative().optional(),
  sfxGain: z.number().nonnegative().optional()
});

export const WorkerRequestSchema = z.discriminatedUnion("command", [
  ExtractBookTextRequestSchema,
  DetectChaptersRequestSchema,
  AnalyzeChapterRequestSchema,
  TranscribeChapterAudioRequestSchema,
  PlanChapterAudioRequestSchema,
  SynthesizeSegmentAudioRequestSchema,
  GenerateAudioAssetsRequestSchema,
  AssembleChapterAudioRequestSchema,
  MixChapterAudioRequestSchema
]);

export const WorkerStatusSchema = z.enum([
  "pending",
  "running",
  "succeeded",
  "failed",
  "skipped",
  "needs_review"
]);

export const WorkerArtifactSchema = z.object({
  kind: z.string().min(1),
  path: z.string().min(1),
  metadata: z.record(z.unknown()).optional()
});

export const WorkerVoiceSchema = z.object({
  id: z.string().min(1),
  displayName: z.string().min(1),
  backend: z.string().min(1)
});

export const WorkerErrorSchema = z.object({
  code: z.string().min(1),
  message: z.string().min(1),
  details: z.record(z.unknown()).optional()
});

export const WorkerResponseSchema = z.object({
  status: WorkerStatusSchema,
  warnings: z.array(z.string()).default([]),
  artifacts: z.array(WorkerArtifactSchema).default([]),
  voices: z.array(WorkerVoiceSchema).default([]),
  error: WorkerErrorSchema.optional()
});

export type ExtractBookTextRequest = z.infer<typeof ExtractBookTextRequestSchema>;
export type DetectChaptersRequest = z.infer<typeof DetectChaptersRequestSchema>;
export type AnalyzeChapterRequest = z.infer<typeof AnalyzeChapterRequestSchema>;
export type TranscribeChapterAudioRequest = z.infer<typeof TranscribeChapterAudioRequestSchema>;
export type PlanChapterAudioRequest = z.infer<typeof PlanChapterAudioRequestSchema>;
export type SynthesizeSegmentAudioRequest = z.infer<typeof SynthesizeSegmentAudioRequestSchema>;
export type GenerateAudioAssetsRequest = z.infer<typeof GenerateAudioAssetsRequestSchema>;
export type AssembleChapterAudioRequest = z.infer<typeof AssembleChapterAudioRequestSchema>;
export type MixChapterAudioRequest = z.infer<typeof MixChapterAudioRequestSchema>;
export type WorkerRequest = z.infer<typeof WorkerRequestSchema>;
export type WorkerResponse = z.infer<typeof WorkerResponseSchema>;
export type WorkerVoice = z.infer<typeof WorkerVoiceSchema>;
