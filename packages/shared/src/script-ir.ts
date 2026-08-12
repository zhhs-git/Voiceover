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
  voiceDesign: z.string().optional(),
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

export const AudioModelSchema = z.enum(["sm-music", "sm-sfx"]);
export const AudioTimingSchema = z.enum(["before", "during", "after"]);
export const MusicVariantLevelSchema = z.enum(["low", "medium", "high"]);

export const MusicPlanSchema = z.object({
  model: z.literal("sm-music"),
  durationSeconds: z.number().positive().max(120),
  prompt: z.string().min(1),
  negativePrompt: z.string().default(""),
  reasonZh: z.string().default("")
});

export const MusicPaletteSchema = z.record(z.string());

export const MusicVariantPlanSchema = z.object({
  id: z.string().min(1),
  level: MusicVariantLevelSchema,
  model: z.literal("sm-music"),
  durationSeconds: z.number().positive().max(120),
  prompt: z.string().min(1),
  negativePrompt: z.string().default(""),
  reasonZh: z.string().default("")
});

export const MusicCuePlanSchema = z
  .object({
    id: z.string().min(1),
    startSegmentIndex: z.number().int().nonnegative(),
    endSegmentIndex: z.number().int().nonnegative(),
    variantId: z.string().min(1),
    reasonZh: z.string().default("")
  })
  .refine((cue) => cue.endSegmentIndex >= cue.startSegmentIndex, {
    message: "endSegmentIndex must be greater than or equal to startSegmentIndex",
    path: ["endSegmentIndex"]
  });

export const MusicBreakPlanSchema = z.object({
  afterSegmentIndex: z.number().int().nonnegative(),
  durationSeconds: z.number().min(2).max(6),
  reasonZh: z.string().default("")
});

export const SfxPlanSchema = z.object({
  id: z.string().min(1),
  model: z.literal("sm-sfx"),
  anchorSegmentIndex: z.number().int().nonnegative(),
  anchorText: z.string().default(""),
  timing: AudioTimingSchema,
  eventZh: z.string().default(""),
  durationSeconds: z.number().positive().max(120),
  prompt: z.string().min(1),
  negativePrompt: z.string().default(""),
  reasonZh: z.string().default("")
});

export const AudioScenePlanSchema = z
  .object({
    id: z.string().min(1),
    startSegmentIndex: z.number().int().nonnegative(),
    endSegmentIndex: z.number().int().nonnegative(),
    summaryZh: z.string().default(""),
    energyArc: z.string().default(""),
    music: MusicPlanSchema.nullable().default(null),
    musicPalette: MusicPaletteSchema.default({}),
    musicVariants: z.array(MusicVariantPlanSchema).default([]),
    musicCues: z.array(MusicCuePlanSchema).default([]),
    musicBreaks: z.array(MusicBreakPlanSchema).default([]),
    sfx: z.array(SfxPlanSchema).default([])
  })
  .refine((scene) => scene.endSegmentIndex >= scene.startSegmentIndex, {
    message: "endSegmentIndex must be greater than or equal to startSegmentIndex",
    path: ["endSegmentIndex"]
  })
  .superRefine((scene, ctx) => {
    for (const [index, effect] of scene.sfx.entries()) {
      if (
        effect.anchorSegmentIndex < scene.startSegmentIndex ||
        effect.anchorSegmentIndex > scene.endSegmentIndex
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "anchorSegmentIndex must be inside the scene range",
          path: ["sfx", index, "anchorSegmentIndex"]
        });
      }
    }

    const variantIds = new Set<string>();
    for (const [index, variant] of scene.musicVariants.entries()) {
      if (variantIds.has(variant.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music variant ids must be unique inside a scene",
          path: ["musicVariants", index, "id"]
        });
      }
      variantIds.add(variant.id);
    }

    const cueIds = new Set<string>();
    let previousCueEnd = scene.startSegmentIndex - 1;
    for (const [index, cue] of [...scene.musicCues]
      .sort((left, right) => left.startSegmentIndex - right.startSegmentIndex)
      .entries()) {
      if (cueIds.has(cue.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music cue ids must be unique inside a scene",
          path: ["musicCues", index, "id"]
        });
      }
      cueIds.add(cue.id);
      if (
        cue.startSegmentIndex < scene.startSegmentIndex ||
        cue.endSegmentIndex > scene.endSegmentIndex
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music cue range must be inside the scene range",
          path: ["musicCues", index]
        });
      }
      if (!variantIds.has(cue.variantId)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music cue must reference a variant declared in the same scene",
          path: ["musicCues", index, "variantId"]
        });
      }
      if (cue.startSegmentIndex <= previousCueEnd) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music cue ranges must not overlap",
          path: ["musicCues", index]
        });
      }
      previousCueEnd = Math.max(previousCueEnd, cue.endSegmentIndex);
    }

    const breakIndices = new Set<number>();
    for (const [index, musicBreak] of scene.musicBreaks.entries()) {
      if (
        musicBreak.afterSegmentIndex < scene.startSegmentIndex ||
        musicBreak.afterSegmentIndex >= scene.endSegmentIndex
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music break must occur strictly inside the scene range",
          path: ["musicBreaks", index, "afterSegmentIndex"]
        });
      }
      if (breakIndices.has(musicBreak.afterSegmentIndex)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "music breaks must use unique afterSegmentIndex values",
          path: ["musicBreaks", index, "afterSegmentIndex"]
        });
      }
      breakIndices.add(musicBreak.afterSegmentIndex);
    }
  });

export const ChapterAudioPlanSchema = z.object({
  version: z.number().int().positive().default(1),
  scenes: z.array(AudioScenePlanSchema).default([])
});

export const ScriptSegmentSchema = z
  .object({
    id: z.string().min(1),
    type: SegmentTypeSchema,
    text: z.string(),
    speakerId: z.string().min(1),
    voiceId: z.string().min(1),
    fallbackVoiceId: z.string().optional(),
    voiceDesign: z.string().optional(),
    voiceDescription: z.string().optional(),
    sceneId: z.string().optional(),
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
  segments: z.array(ScriptSegmentSchema).min(1),
  audioPlan: ChapterAudioPlanSchema.default({ scenes: [] })
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
export type MusicPlan = z.infer<typeof MusicPlanSchema>;
export type MusicPalette = z.infer<typeof MusicPaletteSchema>;
export type MusicVariantLevel = z.infer<typeof MusicVariantLevelSchema>;
export type MusicVariantPlan = z.infer<typeof MusicVariantPlanSchema>;
export type MusicCuePlan = z.infer<typeof MusicCuePlanSchema>;
export type MusicBreakPlan = z.infer<typeof MusicBreakPlanSchema>;
export type SfxPlan = z.infer<typeof SfxPlanSchema>;
export type AudioScenePlan = z.infer<typeof AudioScenePlanSchema>;
export type ChapterAudioPlan = z.infer<typeof ChapterAudioPlanSchema>;
export type ScriptSegment = z.infer<typeof ScriptSegmentSchema>;
export type ChapterScript = z.infer<typeof ChapterScriptSchema>;
export type BookScript = z.infer<typeof BookScriptSchema>;
