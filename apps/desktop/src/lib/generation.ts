import { workerCall } from "./workerCall";

import type { WorkerResponse } from "@audiobook-generator/shared";

type WorkerCall = (
  command: string,
  input: Record<string, unknown>,
) => Promise<WorkerResponse & Record<string, unknown>>;

interface SynthesizeChapterInput {
  scriptPath: string;
  segmentAudioDirectory: string;
  outputPath: string;
  cacheSegments?: boolean;
  worker?: WorkerCall;
}

export async function synthesizeChapter({
  scriptPath,
  segmentAudioDirectory,
  outputPath,
  cacheSegments = true,
  worker = workerCall,
}: SynthesizeChapterInput): Promise<Record<string, unknown>> {
  const synthesis = await worker("synthesize_chapter_audio", {
    scriptPath,
    outputDirectory: segmentAudioDirectory,
    backend: "mimo",
    modelId: "mimo-v2.5-tts-voicedesign",
    mergeSegments: true,
    cacheSegments,
  });

  if (synthesis.status !== "succeeded") {
    return synthesis;
  }

  return worker("assemble_chapter_audio", {
    scriptPath,
    segmentAudioDirectory,
    outputPath,
    mergeSegments: true,
  });
}
