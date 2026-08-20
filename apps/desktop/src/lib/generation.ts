import { workerCall } from "./workerCall";
import type { ModelSettings } from "./modelSettings";

import type { WorkerResponse } from "@audiobook-generator/shared";

type WorkerCall = (
  command: string,
  input: Record<string, unknown>,
) => Promise<WorkerResponse & Record<string, unknown>>;

interface SynthesizeChapterInput {
  scriptPath: string;
  segmentAudioDirectory: string;
  outputPath: string;
  voiceProfileDirectory?: string;
  narratorVoiceId?: "narrator_female" | "narrator_male" | "narrator_default";
  cacheSegments?: boolean;
  tts: Pick<ModelSettings, "ttsBackend" | "ttsModelId">;
  worker?: WorkerCall;
}

interface MixChapterAudioInput {
  bookId: string;
  chapterId: string;
  scriptPath: string;
  segmentAudioDirectory: string;
  voiceAudioPath: string;
  audioAssetsDirectory: string;
  outputPath: string;
  narratorVoiceId?: "narrator_female" | "narrator_male" | "narrator_default";
  voiceGain?: number;
  tts: Pick<ModelSettings, "ttsBackend" | "ttsModelId">;
  worker?: WorkerCall;
}

export async function synthesizeChapter({
  scriptPath,
  segmentAudioDirectory,
  outputPath,
  voiceProfileDirectory,
  narratorVoiceId,
  cacheSegments = true,
  tts,
  worker = workerCall,
}: SynthesizeChapterInput): Promise<Record<string, unknown>> {
  const input: Record<string, unknown> = {
    scriptPath,
    outputDirectory: segmentAudioDirectory,
    backend: tts.ttsBackend,
    modelId: tts.ttsModelId,
    mergeSegments: true,
    cacheSegments,
    mixedOutputPath: outputPath.replace(/\.wav$/i, "_mixed.wav"),
  };
  if (voiceProfileDirectory) input.voiceProfileDirectory = voiceProfileDirectory;
  if (narratorVoiceId) input.narratorVoiceId = narratorVoiceId;
  const synthesis = await worker("synthesize_chapter_audio", input);

  if (synthesis.status !== "succeeded") {
    return synthesis;
  }

  const assemblyInput: Record<string, unknown> = {
    scriptPath,
    segmentAudioDirectory,
    outputPath,
    backend: tts.ttsBackend,
    modelId: tts.ttsModelId,
    mergeSegments: true,
  };
  if (narratorVoiceId) assemblyInput.narratorVoiceId = narratorVoiceId;
  return worker("assemble_chapter_audio", assemblyInput);
}

export async function mixChapterAudio({
  bookId,
  chapterId,
  scriptPath,
  segmentAudioDirectory,
  voiceAudioPath,
  audioAssetsDirectory,
  outputPath,
  narratorVoiceId,
  voiceGain,
  tts,
  worker = workerCall,
}: MixChapterAudioInput): Promise<Record<string, unknown>> {
  const input: Record<string, unknown> = {
    bookId,
    chapterId,
    scriptPath,
    segmentAudioDirectory,
    voiceAudioPath,
    audioAssetsDirectory,
    outputPath,
    mergeSegments: true,
    backend: tts.ttsBackend,
    modelId: tts.ttsModelId,
  };
  if (narratorVoiceId) input.narratorVoiceId = narratorVoiceId;
  if (voiceGain !== undefined) input.voiceGain = voiceGain;
  return worker("mix_chapter_audio", input);
}
