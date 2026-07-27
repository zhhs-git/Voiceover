import { describe, expect, test } from "vitest";
import { ChapterScriptSchema } from "./script-ir";

describe("ChapterScriptSchema", () => {
  test("accepts narration and dialogue segments with voice, emotion, and confidence", () => {
    const parsed = ChapterScriptSchema.parse({
      bookId: "book_123",
      chapterId: "chapter_001",
      title: "Chapter 1",
      language: "en",
      characters: [
        {
          id: "elizabeth",
          canonicalName: "Elizabeth",
          aliases: ["Lizzy"],
          gender: "female",
          ageClass: "adult",
          speakingStyle: "warm and direct",
          voiceId: "character_auto_0123456789abcdef",
          voiceSource: "auto",
          fallbackVoiceId: "female_adult_01",
          voiceDescription: "角色专属的稳定音色。",
          confidence: 0.91
        }
      ],
      voices: [
        {
          id: "character_auto_0123456789abcdef",
          displayName: "角色自动音色（身份生成）",
          genderPresentation: "female",
          ageClass: "adult",
          languages: ["en"],
          styles: ["neutral", "afraid"],
          backend: "mock",
          licenseNotes: "test fixture"
        }
      ],
      segments: [
        {
          id: "seg_0001",
          type: "narration",
          text: "She opened the door slowly.",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "tense",
          intensity: 0.4,
          pace: "slow",
          confidence: 0.93,
          source: { startOffset: 0, endOffset: 29 },
          warnings: []
        },
        {
          id: "seg_0002",
          type: "dialogue",
          text: "Who's there?",
          speakerId: "elizabeth",
          voiceId: "character_auto_0123456789abcdef",
          fallbackVoiceId: "female_adult_01",
          voiceDescription: "角色专属的稳定音色。",
          emotion: "afraid",
          intensity: 0.7,
          pace: "normal",
          confidence: 0.82,
          source: { startOffset: 30, endOffset: 42 },
          warnings: ["speaker_inferred"]
        }
      ]
    });

    expect(parsed.segments).toHaveLength(2);
    expect(parsed.segments[1].speakerId).toBe("elizabeth");
  });
});
