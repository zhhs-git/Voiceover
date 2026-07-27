export const VOICE_PREVIEW_BACKEND = "mimo";
export const VOICE_PREVIEW_MODEL_ID = "mimo-v2.5-tts-voicedesign";
export const VOICE_PREVIEW_TEXT = "你好，这是一段音色预览。";

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
    speakerId: voiceId === "narrator_default" ? "narrator" : "preview",
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
) {
  return {
    scriptPath,
    segmentId,
    outputDirectory,
    backend: VOICE_PREVIEW_BACKEND,
    modelId: VOICE_PREVIEW_MODEL_ID,
  };
}
