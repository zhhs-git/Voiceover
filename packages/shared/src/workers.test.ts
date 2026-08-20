import { describe, expect, test } from "vitest";
import {
  AnalyzeChapterRequestSchema,
  AssembleChapterAudioRequestSchema,
  DetectChaptersRequestSchema,
  ExtractBookTextRequestSchema,
  GenerateAudioAssetsRequestSchema,
  MixChapterAudioRequestSchema,
  PlanChapterAudioRequestSchema,
  SynthesizeChapterAudioRequestSchema,
  SynthesizeSegmentAudioRequestSchema,
  TranscribeChapterAudioRequestSchema,
  WorkerResponseSchema
} from "./workers";

describe("worker protocol schemas", () => {
  test("accepts extract book text requests", () => {
    const parsed = ExtractBookTextRequestSchema.parse({
      command: "extract_book_text",
      bookId: "book_123",
      inputPath: "/tmp/book.epub",
      outputDirectory: "/tmp/book/extracted"
    });

    expect(parsed.command).toBe("extract_book_text");
  });

  test("accepts detect chapters requests", () => {
    const parsed = DetectChaptersRequestSchema.parse({
      command: "detect_chapters",
      bookId: "book_123",
      extractedTextPath: "/tmp/book/extracted/text.json",
      outputDirectory: "/tmp/book/chapters"
    });

    expect(parsed.command).toBe("detect_chapters");
  });

  test("accepts analyze chapter requests", () => {
    const parsed = AnalyzeChapterRequestSchema.parse({
      command: "analyze_chapter",
      bookId: "book_123",
      chapterId: "chapter_001",
      chapterTextPath: "/tmp/book/chapters/chapter_001.txt",
      characterBiblePath: "/tmp/book/scripts/characters.json",
      outputDirectory: "/tmp/book/scripts",
      resumeFromStage: "delivery"
    });

    expect(parsed.chapterId).toBe("chapter_001");
    expect(parsed.resumeFromStage).toBe("delivery");
  });

  test("accepts post-TTS transcription and audio planning requests", () => {
    const transcript = TranscribeChapterAudioRequestSchema.parse({
      command: "transcribe_chapter_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      voiceAudioPath: "/tmp/book/audio/chapter_001.wav",
      analysisDirectory: "/tmp/book/analysis/chapter_001"
    });
    const plan = PlanChapterAudioRequestSchema.parse({
      command: "plan_chapter_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      transcriptPath: "/tmp/book/analysis/chapter_001/transcript.json",
      chapterTextPath: "/tmp/book/chapters/chapter_001.txt"
    });

    expect(transcript.command).toBe("transcribe_chapter_audio");
    expect(plan.command).toBe("plan_chapter_audio");
  });

  test("accepts synthesize segment audio requests", () => {
    const parsed = SynthesizeSegmentAudioRequestSchema.parse({
      command: "synthesize_segment_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      segmentId: "seg_0001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio/segments",
      modelId: "mimo-v2.5-tts-voiceclone",
      voiceProfileDirectory: "/tmp/book/voice-profiles"
    });

    expect(parsed.segmentId).toBe("seg_0001");
    expect(parsed.modelId).toBe("mimo-v2.5-tts-voiceclone");
    expect(parsed.voiceProfileDirectory).toBe("/tmp/book/voice-profiles");
  });

  test("accepts whole-chapter TTS requests with an explicit backend cache key", () => {
    const parsed = SynthesizeChapterAudioRequestSchema.parse({
      command: "synthesize_chapter_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/segments/chapter_001/voxcpm2",
      backend: "voxcpm2",
      modelId: "VoxCPM2",
      voiceProfileDirectory: "/tmp/book/voice-profiles/voxcpm2",
      cacheSegments: true,
      mergeSegments: true,
    });

    expect(parsed.backend).toBe("voxcpm2");
    expect(parsed.modelId).toBe("VoxCPM2");
  });

  test("accepts assemble chapter audio requests", () => {
    const parsed = AssembleChapterAudioRequestSchema.parse({
      command: "assemble_chapter_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      segmentAudioDirectory: "/tmp/book/audio/segments",
      outputPath: "/tmp/book/audio/chapters/chapter_001.mp3"
    });

    expect(parsed.outputPath.endsWith(".mp3")).toBe(true);
  });

  test("accepts Stable Audio asset generation requests", () => {
    const parsed = GenerateAudioAssetsRequestSchema.parse({
      command: "generate_audio_assets",
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio-assets/chapter_001",
      force: false
    });

    expect(parsed.command).toBe("generate_audio_assets");
    expect(parsed.force).toBe(false);
  });

  test("accepts chapter mixing requests", () => {
    const parsed = MixChapterAudioRequestSchema.parse({
      command: "mix_chapter_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      segmentAudioDirectory: "/tmp/book/segments/chapter_001",
      voiceAudioPath: "/tmp/book/audio/chapter_001.wav",
      audioAssetsDirectory: "/tmp/book/audio-assets/chapter_001",
      outputPath: "/tmp/book/audio/chapter_001_mixed.wav",
      gapSeconds: 0.5
    });

    expect(parsed.command).toBe("mix_chapter_audio");
    expect(parsed.outputPath.endsWith("_mixed.wav")).toBe(true);
  });

  test("requires worker responses to include status, warnings, artifacts, and optional error", () => {
    const parsed = WorkerResponseSchema.parse({
      status: "failed",
      warnings: ["low_ocr_confidence"],
      artifacts: [{ kind: "log", path: "/tmp/book/logs/extract.log" }],
      error: {
        code: "ocr_backend_not_configured",
        message: "OCR backend is not configured"
      }
    });

    expect(parsed.error?.code).toBe("ocr_backend_not_configured");
  });

  test("preserves a dynamic voice catalog in worker responses", () => {
    const parsed = WorkerResponseSchema.parse({
      status: "succeeded",
      warnings: [],
      artifacts: [],
      voices: [
        { id: "narrator_default", displayName: "Narrator", backend: "mimo" },
        { id: "female_adult_05", displayName: "Female — Elegant", backend: "mimo" },
        { id: "male_adult_04", displayName: "Male — Strong", backend: "mimo" },
        { id: "neutral_dialogue_01", displayName: "Neutral", backend: "mimo" },
        { id: "narrator_female", displayName: "Narrator — Female", backend: "mimo" },
      ],
    });

    expect(parsed.voices).toHaveLength(5);
    expect(parsed.voices[1].id).toBe("female_adult_05");
  });
});
