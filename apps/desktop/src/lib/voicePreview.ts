import type { ModelSettings } from "./modelSettings";

export const VOICE_PREVIEW_TEXT = "你好，这是一段音色预览。";

function isNarratorVoice(voiceId: string): boolean {
  return ["narrator_default", "narrator_female", "narrator_male"].includes(voiceId);
}

interface VoicePreviewOptions {
  voiceDescription?: string;
  fallbackVoiceId?: string;
}

export function buildVoicePreviewScript(
  bookId: string,
  voiceId: string,
  options: VoicePreviewOptions = {},
) {
  const segmentId = `preview_${voiceId}`;
  const segment = {
    id: segmentId,
    text: VOICE_PREVIEW_TEXT,
    speakerId: isNarratorVoice(voiceId) ? "narrator" : "preview",
    voiceId,
    emotion: "neutral",
    intensity: 0.2,
    pace: "normal",
    ...(options.voiceDescription
      ? { voiceDescription: options.voiceDescription }
      : {}),
    ...(options.fallbackVoiceId
      ? { fallbackVoiceId: options.fallbackVoiceId }
      : {}),
  };
  return {
    bookId,
    chapterId: "voice_preview",
    segments: [segment],
  };
}

export function buildVoicePreviewRequest(
  scriptPath: string,
  outputDirectory: string,
  segmentId: string,
  tts: Pick<ModelSettings, "ttsBackend" | "ttsModelId"> & {
    voiceProfileDirectory?: string;
  },
) {
  const request = {
    scriptPath,
    segmentId,
    outputDirectory,
    backend: tts.ttsBackend,
    modelId: tts.ttsModelId,
    ...(tts.voiceProfileDirectory
      ? { voiceProfileDirectory: tts.voiceProfileDirectory }
      : {}),
  };
  return request;
}
