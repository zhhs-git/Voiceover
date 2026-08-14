import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePipelineStore } from "../state/pipelineStore";

const batchGeneration = vi.hoisted(() => ({
  cancelBatchGeneration: vi.fn(),
  getActiveBatchGeneration: vi.fn(),
  getBatchGenerationStatus: vi.fn(),
  startBatchGeneration: vi.fn(),
}));

vi.mock("../lib/batchGeneration", () => ({
  batchChapterDisplayStage: (chapter: { currentStage?: string | null; nextStage?: string | null }) => {
    if (chapter.currentStage) return chapter.currentStage;
    return chapter.nextStage === "voice_synthesize" || chapter.nextStage === "voice_assemble"
      ? "voice"
      : chapter.nextStage ?? null;
  },
  batchErrorMessage: () => "批量生成失败。",
  cancelBatchGeneration: batchGeneration.cancelBatchGeneration,
  getActiveBatchGeneration: batchGeneration.getActiveBatchGeneration,
  getBatchGenerationStatus: batchGeneration.getBatchGenerationStatus,
  isBatchChapterStageInProgress: (chapter: {
    status: string;
    currentStage?: string | null;
    stageState?: string | null;
  }) => chapter.stageState === "running"
    || (chapter.stageState == null && chapter.status === "running" && Boolean(chapter.currentStage)),
  isBatchChapterWaitingForMiMo: (chapter: {
    status: string;
    nextStage?: string | null;
    stageState?: string | null;
  }) => chapter.status !== "succeeded"
    && chapter.status !== "failed"
    && chapter.status !== "cancelled"
    && chapter.nextStage === "voice_synthesize"
    && chapter.stageState === "ready",
  isActiveBatchGeneration: (status: string) => status === "queued" || status === "running",
  startBatchGeneration: batchGeneration.startBatchGeneration,
}));

import { useGeneration } from "./useGeneration";

const book = {
  title: "测试书籍",
  bookId: "book_1",
  workDir: "/books/book_1",
  chapters: [{
    id: "chapter_1",
    title: "第一章",
    textLength: 10,
    textPath: "/books/book_1/chapters/chapter_1.txt",
  }],
};

describe("useGeneration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    usePipelineStore.getState().resetPipeline();
    usePipelineStore.getState().activateBook(book.bookId);
    batchGeneration.getActiveBatchGeneration.mockResolvedValue(null);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("publishes a terminal batch result into audio paths, assets, and workflow state", async () => {
    batchGeneration.startBatchGeneration.mockResolvedValue({
      batchId: "batch_1",
      bookId: book.bookId,
      status: "succeeded",
      updatedAt: 1_234,
      totalCount: 1,
      completedCount: 1,
      succeededCount: 1,
      chapters: [{
        chapterId: "chapter_1",
        title: "第一章",
        position: 0,
        status: "succeeded",
        voiceAudioPath: "/books/book_1/audio/chapter_1.wav",
        mixedAudioPath: "/books/book_1/audio/chapter_1_mixed.wav",
        audioAssets: [{
          kind: "stable_audio_music",
          path: "/books/book_1/audio-assets/chapter_1/music/scene_1.wav",
          metadata: { assetId: "scene_1", sceneId: "scene_1" },
        }],
      }],
    });
    const setCurrentStep = vi.fn();
    const store = usePipelineStore.getState();
    const { result, unmount } = renderHook(() => useGeneration({
      book,
      analysis: { characters: [], voices: [], scriptPaths: { chapter_1: "/books/book_1/scripts/chapter_1.json" } },
      selectedChapters: new Set(["chapter_1"]),
      chapterAudioPaths: {},
      correctionState: { affectedChapters: [] },
      setStage: store.setStage,
      setError: store.setError,
      setSavedMessage: store.setSavedMessage,
      setAnalyzeProgress: store.setAnalyzeProgress,
      setProgressDetail: store.setProgressDetail,
      setProgress: store.setProgress,
      setChapterAudioPaths: store.setChapterAudioPaths,
      setChapterMixedAudioPaths: store.setChapterMixedAudioPaths,
      setAudioAssets: store.setAudioAssets,
      setGenerationBatch: store.setGenerationBatch,
      setWorkflowStatuses: store.setWorkflowStatuses,
      setCurrentStep,
      abortRef: { current: null },
    }));

    await act(async () => {
      await result.current.handleGenerate();
    });

    const state = usePipelineStore.getState();
    expect(state.chapterAudioPaths.chapter_1).toBe("/books/book_1/audio/chapter_1.wav");
    expect(state.chapterMixedAudioPaths.chapter_1).toBe("/books/book_1/audio/chapter_1_mixed.wav");
    expect(state.audioAssets.chapter_1).toMatchObject([{
      assetId: "scene_1",
      kind: "music",
    }]);
    expect(state.workflows.generation.chapter_1?.status).toBe("succeeded");
    expect(setCurrentStep).toHaveBeenCalledWith("done");
    unmount();
  });

  it("keeps polling after a queued update re-renders the page with a fresh callback prop", async () => {
    vi.useFakeTimers();
    const queuedChapter = {
      chapterId: "chapter_1",
      title: "第一章",
      position: 0,
      status: "running" as const,
      currentStage: "voice",
    };
    batchGeneration.startBatchGeneration.mockResolvedValue({
      batchId: "batch_1",
      bookId: book.bookId,
      status: "running",
      totalCount: 1,
      completedCount: 0,
      chapters: [queuedChapter],
    });
    batchGeneration.getBatchGenerationStatus
      .mockResolvedValueOnce({
        batchId: "batch_1",
        bookId: book.bookId,
        status: "running",
        totalCount: 1,
        completedCount: 0,
        chapters: [queuedChapter],
      })
      .mockResolvedValueOnce({
        batchId: "batch_1",
        bookId: book.bookId,
        status: "succeeded",
        updatedAt: 2_000,
        totalCount: 1,
        completedCount: 1,
        succeededCount: 1,
        chapters: [{
          ...queuedChapter,
          status: "succeeded",
          currentStage: null,
          voiceAudioPath: "/books/book_1/audio/chapter_1.wav",
          mixedAudioPath: "/books/book_1/audio/chapter_1_mixed.wav",
        }],
      });
    usePipelineStore.getState().setSelectedChapters(new Set(["chapter_1"]), book.bookId);
    const stableAnalysis = {
      characters: [],
      voices: [],
      scriptPaths: { chapter_1: "/books/book_1/scripts/chapter_1.json" },
    };
    const { result, unmount } = renderHook(() => {
      // Subscribe to the store like BookDetailView does. A batch publication
      // now re-renders this hook before its next poll timer fires.
      const pipeline = usePipelineStore();
      return useGeneration({
        book,
        analysis: stableAnalysis,
        selectedChapters: pipeline.selectedChapters,
        chapterAudioPaths: pipeline.chapterAudioPaths,
        correctionState: { affectedChapters: [] },
        setStage: pipeline.setStage,
        setError: pipeline.setError,
        setSavedMessage: pipeline.setSavedMessage,
        setAnalyzeProgress: pipeline.setAnalyzeProgress,
        setProgressDetail: pipeline.setProgressDetail,
        setProgress: pipeline.setProgress,
        setChapterAudioPaths: pipeline.setChapterAudioPaths,
        setChapterMixedAudioPaths: pipeline.setChapterMixedAudioPaths,
        setAudioAssets: pipeline.setAudioAssets,
        setGenerationBatch: pipeline.setGenerationBatch,
        setWorkflowStatuses: pipeline.setWorkflowStatuses,
        // Intentionally new on every Zustand-driven render. The hook must not
        // use this callback identity as a reason to clear its poll timer.
        setCurrentStep: () => {},
        abortRef: { current: null },
      });
    });

    await act(async () => {
      await result.current.handleGenerate();
      await Promise.resolve();
    });
    expect(batchGeneration.getBatchGenerationStatus).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_200);
    });

    expect(batchGeneration.getBatchGenerationStatus).toHaveBeenCalledTimes(2);
    expect(usePipelineStore.getState().workflows.generation.chapter_1?.status).toBe("succeeded");
    unmount();
  });
});
