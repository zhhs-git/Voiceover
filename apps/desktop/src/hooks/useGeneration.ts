import { useCallback } from "react";
import { flushSync } from "react-dom";
import { invoke } from "../lib/platform";

import type {
  AnalysisState,
  BookState,
  ChapterMeta,
  PipelineStage,
  ProgressDetail,
  WorkspaceStep,
} from "../types";
import { generationProgressDetails } from "../lib/generationProgress";
import { synthesizeChapter } from "../lib/generation";

interface UseGenerationDeps {
  book: BookState | null;
  analysis: AnalysisState | null;
  selectedChapters: Set<string>;
  chapterAudioPaths: Record<string, string>;
  correctionState: { affectedChapters: string[]; dirty?: boolean };
  setStage: (stage: PipelineStage, owner?: string) => void;
  setError: (error: string | null, owner?: string) => void;
  setAnalyzeProgress: (msg: string, owner?: string) => void;
  setProgressDetail: (details: ProgressDetail[], owner?: string) => void;
  setProgress: (progress: number, owner?: string) => void;
  setChapterAudioPaths: (
    paths:
      | Record<string, string>
      | ((prev: Record<string, string>) => Record<string, string>),
    owner?: string,
  ) => void;
  setCurrentStep: (step: WorkspaceStep) => void;
  abortRef: React.MutableRefObject<AbortController | null>;
}

export function useGeneration(deps: UseGenerationDeps) {
  const {
    book,
    analysis,
    selectedChapters,
    chapterAudioPaths,
    correctionState,
    setStage,
    setError,
    setAnalyzeProgress,
    setProgressDetail,
    setProgress,
    setChapterAudioPaths,
    setCurrentStep,
    abortRef,
  } = deps;

  const generateChapters = useCallback(
    async (chaptersToGenerate: ChapterMeta[], cacheSegments = true) => {
      if (!book || !analysis) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setStage("generating", book.bookId);
      setError(null, book.bookId);
      setAnalyzeProgress("", book.bookId);

      setProgressDetail([
        { label: "后端", value: "MiMo V2.5 TTS 音色设计" },
        { label: "章节", value: String(chaptersToGenerate.length) },
      ], book.bookId);

      const startTime = Date.now();
      let totalSegments = 0;
      let doneSegments = 0;
      const newAudioPaths: Record<string, string> = {};

      try {
        for (let ci = 0; ci < chaptersToGenerate.length; ci++) {
          if (controller.signal.aborted) break;
          const chapter = chaptersToGenerate[ci];
          const scriptPath = analysis.scriptPaths[chapter.id];
          if (!scriptPath) continue;

          const segDir = `${book.workDir}/segments/${chapter.id}`;
          const assembledPath = `${book.workDir}/audio/${chapter.id}.wav`;

          const scriptRaw = await invoke<string>("run_worker", {
            command: "_read_file",
            inputJson: JSON.stringify({ path: scriptPath }),
          }).catch(() => "{}");

          const script = JSON.parse(scriptRaw) as {
            segments?: Array<{
              id: string;
              voiceId?: string;
              emotion?: string;
            }>;
          };
          const segments = script.segments ?? [];

          setAnalyzeProgress(
            `正在合成第 ${ci + 1} 章（共 ${chaptersToGenerate.length} 章，${segments.length} 个片段）…`,
            book.bookId,
          );
          totalSegments += segments.length;

          const setGenerationProgress = () => {
            setProgress(
              40 +
                Math.round((doneSegments / Math.max(totalSegments, 1)) * 50),
              book.bookId,
            );
            setProgressDetail(
              generationProgressDetails({
                now: Date.now(),
                startTime,
                doneSegments,
                totalSegments,
                chapterIndex: ci + 1,
                chapterCount: chaptersToGenerate.length,
                segmentCount: segments.length,
              }),
              book.bookId,
            );
          };

          flushSync(() => {
            setGenerationProgress();
          });

          const progressTimer = window.setInterval(setGenerationProgress, 2000);
          let result: Record<string, unknown>;
          try {
            result = await synthesizeChapter({
              scriptPath,
              segmentAudioDirectory: segDir,
              outputPath: assembledPath,
              cacheSegments,
            });
          } finally {
            window.clearInterval(progressTimer);
          }

          if (result.status !== "succeeded") {
            const workerError = result.error as { message?: unknown } | undefined;
            const message =
              typeof workerError?.message === "string"
                ? workerError.message
                : `章节《${chapter.title}》音频生成失败。`;
            throw new Error(message);
          }
          doneSegments += segments.length;
          newAudioPaths[chapter.id] = assembledPath;
        }

        setChapterAudioPaths((prev) => ({ ...prev, ...newAudioPaths }), book.bookId);
        const wasStopped = controller.signal.aborted;
        setProgress(
          wasStopped
            ? 40 +
                Math.round((doneSegments / Math.max(totalSegments, 1)) * 50)
            : 100,
          book.bookId,
        );
        setAnalyzeProgress(
          wasStopped
            ? "生成已停止，已有部分音频可用。"
            : "音频生成完成。",
          book.bookId,
        );
        const generatedCount = Object.keys(newAudioPaths).length;
        setStage(generatedCount > 0 || !wasStopped ? "done" : "idle", book.bookId);
        if (abortRef.current === controller) abortRef.current = null;
        if (generatedCount > 0) setCurrentStep("done");
      } catch (err) {
        if (Object.keys(newAudioPaths).length > 0) {
          setChapterAudioPaths((prev) => ({ ...prev, ...newAudioPaths }), book.bookId);
        }
        if (!controller.signal.aborted) {
          setError(`音频生成失败：${String(err)}`, book.bookId);
          setAnalyzeProgress("音频生成失败。", book.bookId);
          setStage("error", book.bookId);
          if (abortRef.current === controller) abortRef.current = null;
        } else {
          setAnalyzeProgress("生成已停止。", book.bookId);
          setStage("idle", book.bookId);
          if (abortRef.current === controller) abortRef.current = null;
        }
      }
    },
    [
      book,
      analysis,
      abortRef,
      setAnalyzeProgress,
      setChapterAudioPaths,
      setCurrentStep,
      setError,
      setProgress,
      setProgressDetail,
      setStage,
    ],
  );

  const handleGenerate = useCallback(async () => {
    if (!book || !analysis) return;
    const chaptersToGenerate = (
      correctionState.affectedChapters.length > 0
        ? book.chapters.filter((c) =>
            correctionState.affectedChapters.includes(c.id),
          )
        : book.chapters
    ).filter(
      (c) => selectedChapters.has(c.id) && analysis.scriptPaths[c.id],
    );

    await generateChapters(chaptersToGenerate);
  }, [book, analysis, correctionState.affectedChapters, selectedChapters, generateChapters]);

  const handleRegenerateChapter = useCallback(
    async (chapter: ChapterMeta) => {
      await generateChapters([chapter], false);
    },
    [generateChapters],
  );

  const handleRegenerateAll = useCallback(async () => {
    if (!book) return;
    const generatedChapters = book.chapters.filter(
      (c) => chapterAudioPaths[c.id],
    );
    await generateChapters(generatedChapters, false);
  }, [book, chapterAudioPaths, generateChapters]);

  return { handleGenerate, handleRegenerateChapter, handleRegenerateAll };
}
