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
    expect(parsed.audioPlan.scenes).toEqual([]);
  });

  test("accepts a Stable Audio scene plan and validates SFX anchors", () => {
    const parsed = ChapterScriptSchema.parse({
      bookId: "book_123",
      chapterId: "chapter_001",
      language: "zh",
      segments: [
        {
          id: "seg_0001",
          type: "narration",
          text: "雨落在青石板上。",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "tense",
          intensity: 0.4,
          pace: "normal",
          confidence: 0.9,
          warnings: []
        }
      ],
      audioPlan: {
        scenes: [
          {
            id: "scene_001",
            startSegmentIndex: 0,
            endSegmentIndex: 0,
            summaryZh: "雨夜街道",
            music: {
              model: "sm-music",
              durationSeconds: 30,
              prompt: "Dark historical suspense instrumental bed",
              negativePrompt: "vocals, speech",
              reasonZh: "旁白下的悬疑氛围"
            },
            sfx: [
              {
                id: "sfx_001",
                model: "sm-sfx",
                anchorSegmentIndex: 0,
                timing: "during",
                eventZh: "雨声",
                durationSeconds: 5,
                prompt: "TrackType: SFX, heavy rain on wet stone pavement",
                negativePrompt: "music, speech",
                reasonZh: "强化环境感"
              }
            ]
          }
        ]
      }
    });

    expect(parsed.audioPlan.scenes[0].music?.model).toBe("sm-music");
    expect(parsed.audioPlan.scenes[0]).not.toHaveProperty("emotion");
    expect(parsed.audioPlan.scenes[0]).not.toHaveProperty("narrationDirection");
    expect(parsed.audioPlan.scenes[0].sfx[0].anchorSegmentIndex).toBe(0);

    expect(() => ChapterScriptSchema.parse({
      bookId: "book_123",
      chapterId: "chapter_001",
      language: "zh",
      segments: [
        {
          id: "seg_0001",
          type: "narration",
          text: "雨落在青石板上。",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "tense",
          intensity: 0.4,
          pace: "normal",
          confidence: 0.9,
          warnings: []
        }
      ],
      audioPlan: {
        scenes: [{
          id: "scene_001",
          startSegmentIndex: 0,
          endSegmentIndex: 0,
          music: null,
          sfx: [{
            id: "sfx_001",
            model: "sm-sfx",
            anchorSegmentIndex: 1,
            timing: "during",
            durationSeconds: 5,
            prompt: "TrackType: SFX, rain",
            eventZh: "雨声"
          }]
        }]
      }
    })).toThrow();
  });

  test("accepts the v2 music palette, variants, cues, and intentional breaks", () => {
    const parsed = ChapterScriptSchema.parse({
      bookId: "book_123",
      chapterId: "chapter_002",
      language: "zh",
      segments: [
        {
          id: "seg_0001",
          type: "narration",
          text: "街灯渐渐熄灭。",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "solemn",
          intensity: 0.5,
          pace: "slow",
          confidence: 0.9,
          warnings: []
        },
        {
          id: "seg_0002",
          type: "dialogue",
          text: "我们走吧。",
          speakerId: "traveler",
          voiceId: "traveler_voice",
          emotion: "resolute",
          intensity: 0.6,
          pace: "normal",
          confidence: 0.9,
          warnings: []
        },
        {
          id: "seg_0003",
          type: "narration",
          text: "远处传来钟声。",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "tense",
          intensity: 0.6,
          pace: "normal",
          confidence: 0.9,
          warnings: []
        }
      ],
      audioPlan: {
        version: 2,
        scenes: [
          {
            id: "scene_002",
            startSegmentIndex: 0,
            endSegmentIndex: 2,
            summaryZh: "夜路",
            energyArc: "低能量观察→情绪发展→紧张收束",
            music: null,
            musicPalette: {
              identity: "克制的夜行悬疑",
              instrumentation: "低音弦乐、木管",
              register: "低到中",
              texture: "稀疏、呼吸感",
              tempo: "缓慢脉动",
              reasonZh: "随夜色和紧张感变化"
            },
            musicVariants: [
              {
                id: "scene_002_low",
                level: "low",
                model: "sm-music",
                durationSeconds: 30,
                prompt: "TrackType: Music, VocalType: Instrumental, restrained night texture"
              },
              {
                id: "scene_002_medium",
                level: "medium",
                model: "sm-music",
                durationSeconds: 30,
                prompt: "TrackType: Music, VocalType: Instrumental, guarded pulse"
              },
              {
                id: "scene_002_high",
                level: "high",
                model: "sm-music",
                durationSeconds: 30,
                prompt: "TrackType: Music, VocalType: Instrumental, tense forward motion"
              }
            ],
            musicCues: [
              {
                id: "cue_001",
                startSegmentIndex: 0,
                endSegmentIndex: 1,
                variantId: "scene_002_low",
                reasonZh: "铺底"
              },
              {
                id: "cue_002",
                startSegmentIndex: 2,
                endSegmentIndex: 2,
                variantId: "scene_002_high",
                reasonZh: "钟声后的紧张"
              }
            ],
            musicBreaks: [
              {
                afterSegmentIndex: 1,
                durationSeconds: 4,
                reasonZh: "对白结束后的呼吸"
              }
            ],
            sfx: []
          }
        ]
      }
    });

    expect(parsed.audioPlan.version).toBe(2);
    expect(parsed.audioPlan.scenes[0].energyArc).toBe("低能量观察→情绪发展→紧张收束");
    expect(parsed.audioPlan.scenes[0].musicPalette.identity).toBe("克制的夜行悬疑");
    expect(parsed.audioPlan.scenes[0].musicVariants).toHaveLength(3);
    expect(parsed.audioPlan.scenes[0].musicCues[1].variantId).toBe("scene_002_high");
    expect(parsed.audioPlan.scenes[0].musicBreaks[0].durationSeconds).toBe(4);

    expect(() => ChapterScriptSchema.parse({
      bookId: "book_123",
      chapterId: "chapter_003",
      language: "zh",
      segments: [
        {
          id: "seg_0001",
          type: "narration",
          text: "测试。",
          speakerId: "narrator",
          voiceId: "narrator_default",
          emotion: "neutral",
          intensity: 0,
          pace: "normal",
          confidence: 1,
          warnings: []
        }
      ],
      audioPlan: {
        version: 2,
        scenes: [{
          id: "scene_003",
          startSegmentIndex: 0,
          endSegmentIndex: 0,
          music: null,
          musicVariants: [],
          musicCues: [{
            id: "cue_bad",
            startSegmentIndex: 0,
            endSegmentIndex: 0,
            variantId: "missing_variant"
          }],
          musicBreaks: [],
          sfx: []
        }]
      }
    })).toThrow();
  });
});
