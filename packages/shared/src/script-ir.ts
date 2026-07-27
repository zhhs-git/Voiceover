import { z } from "zod";

export const SegmentTypeSchema = z.enum([
  "narration",
  "dialogue",
  "heading",
  "silence",
  "sound_cue"
]);

export const EmotionSchema = z.enum([
  "neutral",
  "happy",
  "sad",
  "angry",
  "afraid",
  "tense",
  "teasing",
  "whispering",
  "excited",
  "tired",
  "grief",
  "cold",
  "pleading",
  "surprised",
  "gentle",
  "resolute",
  "nervous",
  "contemptuous",
  "solemn",
  "bitter"
]);

export const PaceSchema = z.enum(["slow", "normal", "fast"]);
export const GenderSchema = z.enum(["female", "male", "neutral", "unknown"]);
export const AgeClassSchema = z.enum(["child", "young", "adult", "older", "unknown"]);
export const VoiceSourceSchema = z.enum(["auto", "manual"]);
export const IdentityStatusSchema = z.enum(["provisional", "confirmed", "merged"]);

export const CharacterProfileSchema = z.object({
  id: z.string().min(1),
  canonicalName: z.string().min(1),
  aliases: z.array(z.string()).default([]),
  gender: GenderSchema,
  ageClass: AgeClassSchema,
  identityStatus: IdentityStatusSchema.optional(),
  speakingStyle: z.string().optional(),
  voiceId: z.string().optional(),
  voiceSource: VoiceSourceSchema.optional(),
  voiceAssignmentVersion: z.number().int().positive().optional(),
  voiceProfile: z.string().optional(),
  fallbackVoiceId: z.string().optional(),
  voiceDescription: z.string().optional(),
  confidence: z.number().min(0).max(1)
});

export const VoiceProfileSchema = z.object({
  id: z.string().min(1),
  displayName: z.string().min(1),
  genderPresentation: GenderSchema,
  ageClass: AgeClassSchema,
  languages: z.array(z.string().min(2)).min(1),
  styles: z.array(EmotionSchema).default(["neutral"]),
  backend: z.string().min(1),
  licenseNotes: z.string().optional()
});

export const SourceLocationSchema = z.object({
  startOffset: z.number().int().nonnegative(),
  endOffset: z.number().int().nonnegative()
});

export const ScriptSegmentSchema = z
  .object({
    id: z.string().min(1),
    type: SegmentTypeSchema,
    text: z.string(),
    speakerId: z.string().min(1),
    voiceId: z.string().min(1),
    fallbackVoiceId: z.string().optional(),
    voiceDescription: z.string().optional(),
    emotion: EmotionSchema,
    intensity: z.number().min(0).max(1),
    pace: PaceSchema,
    confidence: z.number().min(0).max(1),
    source: SourceLocationSchema.optional(),
    warnings: z.array(z.string()).default([])
  })
  .refine((segment) => !segment.source || segment.source.endOffset >= segment.source.startOffset, {
    message: "source.endOffset must be greater than or equal to source.startOffset",
    path: ["source", "endOffset"]
  });

export const ChapterScriptSchema = z.object({
  bookId: z.string().min(1),
  chapterId: z.string().min(1),
  title: z.string().optional(),
  language: z.string().min(2),
  characters: z.array(CharacterProfileSchema).default([]),
  voices: z.array(VoiceProfileSchema).default([]),
  segments: z.array(ScriptSegmentSchema).min(1)
});

export const BookScriptSchema = z.object({
  bookId: z.string().min(1),
  title: z.string().optional(),
  sourceLanguage: z.string().min(2),
  outputLanguage: z.string().min(2),
  chapters: z.array(ChapterScriptSchema)
});

export type SegmentType = z.infer<typeof SegmentTypeSchema>;
export type Emotion = z.infer<typeof EmotionSchema>;
export type CharacterProfile = z.infer<typeof CharacterProfileSchema>;
export type VoiceProfile = z.infer<typeof VoiceProfileSchema>;
export type ScriptSegment = z.infer<typeof ScriptSegmentSchema>;
export type ChapterScript = z.infer<typeof ChapterScriptSchema>;
export type BookScript = z.infer<typeof BookScriptSchema>;
