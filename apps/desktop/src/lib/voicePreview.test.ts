import { describe, expect, test } from "vitest";
import {
  buildVoicePreviewRequest,
  buildVoicePreviewScript,
  VOICE_PREVIEW_BACKEND,
  VOICE_PREVIEW_MODEL_ID,
} from "./voicePreview";

describe("voice preview", () => {
  test("uses the active MiMo backend instead of Kokoro", () => {
    expect(
      buildVoicePreviewRequest(
        "/tmp/preview.json",
        "/tmp/previews",
        "preview_female_adult_01",
      ),
    ).toEqual({
      scriptPath: "/tmp/preview.json",
      segmentId: "preview_female_adult_01",
      outputDirectory: "/tmp/previews",
      backend: VOICE_PREVIEW_BACKEND,
      modelId: VOICE_PREVIEW_MODEL_ID,
    });
    expect(VOICE_PREVIEW_BACKEND).toBe("mimo");
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
