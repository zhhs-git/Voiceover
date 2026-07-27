import { describe, expect, test, vi } from "vitest";
import { synthesizeChapter } from "./generation";

describe("synthesizeChapter", () => {
  test("synthesizes a whole chapter in one worker call before assembling", async () => {
    const worker = vi.fn(async (command: string) => {
      if (command === "synthesize_chapter_audio") {
        return { status: "succeeded" as const, artifacts: [], warnings: [], voices: [] };
      }
      if (command === "assemble_chapter_audio") {
        return { status: "succeeded" as const, artifacts: [], warnings: [], voices: [] };
      }
      throw new Error(`unexpected command: ${command}`);
    });

    await synthesizeChapter({
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      segmentAudioDirectory: "/tmp/book/segments/chapter_001",
      outputPath: "/tmp/book/audio/chapter_001.wav",
      worker,
    });

    expect(worker).toHaveBeenCalledTimes(2);
    expect(worker).toHaveBeenNthCalledWith(1, "synthesize_chapter_audio", {
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/segments/chapter_001",
      backend: "mimo",
      modelId: "mimo-v2.5-tts-voicedesign",
      mergeSegments: true,
      cacheSegments: true,
    });
    expect(worker).toHaveBeenNthCalledWith(2, "assemble_chapter_audio", {
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      segmentAudioDirectory: "/tmp/book/segments/chapter_001",
      outputPath: "/tmp/book/audio/chapter_001.wav",
      mergeSegments: true,
    });
  });

  test("does not assemble when segment synthesis fails", async () => {
    const worker = vi.fn(async (command: string) => {
      if (command === "synthesize_chapter_audio") {
        return {
          status: "failed" as const,
          artifacts: [],
          warnings: [],
          voices: [],
          error: { code: "incomplete_segment_audio", message: "missing segment" },
        };
      }
      throw new Error(`unexpected command: ${command}`);
    });

    const result = await synthesizeChapter({
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      segmentAudioDirectory: "/tmp/book/segments/chapter_001",
      outputPath: "/tmp/book/audio/chapter_001.wav",
      worker,
    });

    expect(result.status).toBe("failed");
    expect(worker).toHaveBeenCalledTimes(1);
    expect(worker).toHaveBeenCalledWith("synthesize_chapter_audio", expect.anything());
  });
});
