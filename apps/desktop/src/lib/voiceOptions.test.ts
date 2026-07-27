import { describe, expect, test } from "vitest";
import type { CharacterMeta, VoiceMeta } from "../types";
import { buildVoiceOptions } from "./voiceOptions";

const providerVoices: VoiceMeta[] = [
  { id: "narrator_default", displayName: "Narrator", backend: "mimo" },
  { id: "female_adult_01", displayName: "Female — Warm", backend: "mimo" },
  { id: "female_adult_02", displayName: "Female — Bright", backend: "mimo" },
  { id: "female_adult_03", displayName: "Female — Soft", backend: "mimo" },
  { id: "female_adult_05", displayName: "Female — Elegant", backend: "mimo" },
];

const characters: CharacterMeta[] = [
  {
    id: "elizabeth",
    canonicalName: "Elizabeth",
    aliases: [],
    gender: "female",
    voiceId: "female_adult_05",
    confidence: 0.9,
  },
  {
    id: "old-script-character",
    canonicalName: "Old Script Character",
    aliases: [],
    gender: "female",
    voiceId: "female_british_01",
    confidence: 0.8,
  },
];

const analysisVoices: VoiceMeta[] = [
  {
    id: "female_british_01",
    displayName: "Female — British",
    backend: "kokoro",
  },
];

describe("buildVoiceOptions", () => {
  test("uses the provider catalog dynamically instead of a fixed four-item list", () => {
    const options = buildVoiceOptions({
      providerVoices,
      analysisVoices: [],
      characters: [characters[0]],
    });

    expect(options.map((voice) => voice.id)).toEqual([
      "narrator_default",
      "female_adult_01",
      "female_adult_02",
      "female_adult_03",
      "female_adult_05",
    ]);
    expect(options).toHaveLength(5);
  });

  test("keeps an existing unsupported assignment visible without treating it as selectable", () => {
    const options = buildVoiceOptions({
      providerVoices,
      analysisVoices,
      characters,
    });

    const currentVoice = options.find((voice) => voice.id === "female_british_01");
    expect(currentVoice).toEqual({
      id: "female_british_01",
      displayName: "女声 — 英式明亮",
      available: false,
    });
  });

  test("shows automatic identity voices as the active generated choice", () => {
    const options = buildVoiceOptions({
      providerVoices,
      analysisVoices: [
        {
          id: "character_auto_0123456789abcdef",
          displayName: "角色自动音色（身份生成）",
          backend: "mimo",
        },
      ],
      characters: [
        {
          id: "character_1",
          canonicalName: "角色一",
          aliases: [],
          gender: "female",
          voiceId: "character_auto_0123456789abcdef",
          voiceSource: "auto",
          confidence: 0.9,
        },
      ],
    });

    expect(options.find((voice) => voice.id === "character_auto_0123456789abcdef"))
      .toEqual({
        id: "character_auto_0123456789abcdef",
        displayName: "角色自动音色（身份生成）",
        available: true,
      });
  });
});
