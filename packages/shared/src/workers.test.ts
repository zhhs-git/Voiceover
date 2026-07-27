import { describe, expect, test } from "vitest";
import {
  AnalyzeChapterRequestSchema,
  AssembleChapterAudioRequestSchema,
  DetectChaptersRequestSchema,
  ExtractBookTextRequestSchema,
  SynthesizeSegmentAudioRequestSchema,
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
      outputDirectory: "/tmp/book/scripts"
    });

    expect(parsed.chapterId).toBe("chapter_001");
  });

  test("accepts synthesize segment audio requests", () => {
    const parsed = SynthesizeSegmentAudioRequestSchema.parse({
      command: "synthesize_segment_audio",
      bookId: "book_123",
      chapterId: "chapter_001",
      segmentId: "seg_0001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio/segments"
    });

    expect(parsed.segmentId).toBe("seg_0001");
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
