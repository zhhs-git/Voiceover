import { describe, expect, test } from "vitest";
import {
  buildVoicePreviewRequest,
  buildVoicePreviewScript,
} from "./voicePreview";

describe("voice preview", () => {
  test("uses the selected TTS backend and model", () => {
    expect(
      buildVoicePreviewRequest(
        "/tmp/preview.json",
        "/tmp/previews",
        "preview_female_adult_01",
        {
          ttsBackend: "mimo",
          ttsModelId: "mimo-v2.5-tts-voiceclone",
        },
      ),
    ).toEqual({
      scriptPath: "/tmp/preview.json",
      segmentId: "preview_female_adult_01",
      outputDirectory: "/tmp/previews",
      backend: "mimo",
      modelId: "mimo-v2.5-tts-voiceclone",
    });
  });

  test("uses a Chinese preview phrase and preserves narrator identity", () => {
    const script = buildVoicePreviewScript("book_123", "narrator_default");

    expect(script.segments[0]).toMatchObject({
      text: "你好，这是一段音色预览。",
      speakerId: "narrator",
      voiceId: "narrator_default",
      pace: "normal",
    });
  });

  test("keeps an automatic character's generated voice design in its preview", () => {
    const script = buildVoicePreviewScript(
      "book_123",
      "character_auto_0123456789abcdef",
      {
        voiceDescription: "角色专属的稳定中文声线。",
        fallbackVoiceId: "male_adult_01",
      },
    );

    expect(script.segments[0]).toMatchObject({
      speakerId: "preview",
      voiceId: "character_auto_0123456789abcdef",
      voiceDescription: "角色专属的稳定中文声线。",
      fallbackVoiceId: "male_adult_01",
    });
  });
});
