import { workerCall } from "./workerCall";

type WorkerResult = Record<string, unknown> & { status?: string; error?: unknown };

export async function transcribeChapterAudio(input: {
  bookId: string;
  chapterId: string;
  scriptPath: string;
  voiceAudioPath: string;
  analysisDirectory: string;
  whisperModel?: string;
  whisperPython?: string;
}): Promise<WorkerResult> {
  return workerCall("transcribe_chapter_audio", input);
}

export async function planChapterAudio(input: {
  bookId: string;
  chapterId: string;
  scriptPath: string;
  transcriptPath: string;
  chapterTextPath: string;
  analysisDirectory: string;
}): Promise<WorkerResult> {
  return workerCall("plan_chapter_audio", input);
}
