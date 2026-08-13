import { useCallback, useEffect, useRef } from "react";

import type {
  AnalysisState,
  AudioAsset,
  BookState,
  ChapterWorkflowStatus,
  ChapterMeta,
  PipelineStage,
  ProgressDetail,
  WorkspaceStep,
} from "../types";
import { audioAssetsFromArtifacts } from "../lib/audioAssets";
import {
  batchErrorMessage,
  cancelBatchGeneration,
  getActiveBatchGeneration,
  getBatchGenerationStatus,
  isActiveBatchGeneration,
  startBatchGeneration,
  type BatchGenerationResponse,
} from "../lib/batchGeneration";
import { terminalGenerationWorkflowsFromBatch } from "../lib/workflowStatus";
import { workerCall } from "../lib/workerCall";

interface UseGenerationDeps {
  book: BookState | null;
  analysis: AnalysisState | null;
  selectedChapters: Set<string>;
  chapterAudioPaths: Record<string, string>;
  correctionState: { affectedChapters: string[]; dirty?: boolean };
  setStage: (stage: PipelineStage, owner?: string) => void;
  setError: (error: string | null, owner?: string) => void;
  setSavedMessage: (message: string | null, owner?: string) => void;
  setAnalyzeProgress: (msg: string, owner?: string) => void;
  setProgressDetail: (details: ProgressDetail[], owner?: string) => void;
  setProgress: (progress: number, owner?: string) => void;
  setChapterAudioPaths: (
    paths:
      | Record<string, string>
      | ((prev: Record<string, string>) => Record<string, string>),
    owner?: string,
  ) => void;
  setChapterMixedAudioPaths: (
    paths:
      | Record<string, string>
      | ((prev: Record<string, string>) => Record<string, string>),
    owner?: string,
  ) => void;
  setAudioAssets: (
    assets:
      | Record<string, AudioAsset[]>
      | ((prev: Record<string, AudioAsset[]>) => Record<string, AudioAsset[]>),
    owner?: string,
  ) => void;
  setGenerationBatch: (
    batch:
      | BatchGenerationResponse
      | null
      | ((prev: BatchGenerationResponse | null) => BatchGenerationResponse | null),
    owner?: string,
  ) => void;
  setWorkflowStatuses: (
    kind: "generation",
    statuses:
      | Record<string, ChapterWorkflowStatus>
      | ((prev: Record<string, ChapterWorkflowStatus>) => Record<string, ChapterWorkflowStatus>),
    owner?: string,
  ) => void;
  setCurrentStep: (step: WorkspaceStep) => void;
  abortRef: React.MutableRefObject<AbortController | null>;
}

const STAGE_LABELS: Record<string, string> = {
  voice: "原章节配音",
  transcript: "Whisper 转录",
  audio_plan: "背景音/音效规划",
  stable_audio: "Stable Audio 背景音/音效",
  mix: "最终混音",
};

function stageLabel(stage: string | null | undefined): string {
  return (stage && STAGE_LABELS[stage]) || "准备中";
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
    setSavedMessage,
    setAnalyzeProgress,
    setProgressDetail,
    setProgress,
    setChapterAudioPaths,
    setChapterMixedAudioPaths,
    setAudioAssets,
    setGenerationBatch,
    setWorkflowStatuses,
    setCurrentStep,
  } = deps;
  const activeBatchIdRef = useRef<string | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const requestInFlightRef = useRef(false);
  const activeBookIdRef = useRef<string | null>(null);
  const setCurrentStepRef = useRef(setCurrentStep);
  setCurrentStepRef.current = setCurrentStep;

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const publishBatch = useCallback((response: BatchGenerationResponse, owner: string) => {
    setGenerationBatch(response, owner);
    const total = response.totalCount ?? response.chapters.length;
    const completed = response.completedCount ?? response.chapters.filter((chapter) =>
      ["succeeded", "failed", "cancelled"].includes(chapter.status),
    ).length;
    const current = response.chapters.find((chapter) => chapter.status === "running");
    const title = current?.title || response.chapters.find((chapter) => chapter.chapterId === response.currentChapterId)?.title;
    const stage = current?.currentStage ?? response.currentStage;

    setProgress(total > 0 ? Math.round((completed / total) * 100) : 0, owner);
    setProgressDetail([
      { label: "执行方式", value: "章节顺序队列（MiMo 片段受控并发 / Whisper / LLM / Stable Audio）" },
      { label: "进度", value: `${completed} / ${total} 章` },
      { label: "当前章节", value: title || "等待队列" },
      { label: "当前阶段", value: stageLabel(stage) },
    ], owner);

    setChapterAudioPaths((previous) => {
      const next = { ...previous };
      for (const chapter of response.chapters) {
        if (chapter.voiceAudioPath) next[chapter.chapterId] = chapter.voiceAudioPath;
      }
      return next;
    }, owner);
    setChapterMixedAudioPaths((previous) => {
      const next = { ...previous };
      for (const chapter of response.chapters) {
        if (chapter.mixedAudioPath) next[chapter.chapterId] = chapter.mixedAudioPath;
      }
      return next;
    }, owner);
    setAudioAssets((previous) => {
      const next = { ...previous };
      for (const chapter of response.chapters) {
        if (!chapter.audioAssets) continue;
        next[chapter.chapterId] = audioAssetsFromArtifacts(chapter.audioAssets).map((asset) => ({
          ...asset,
          refreshKey: `${response.updatedAt ?? Date.now()}-${asset.kind}-${asset.assetId}`,
        }));
      }
      return next;
    }, owner);

    if (isActiveBatchGeneration(response.status)) {
      setStage("generating", owner);
      setAnalyzeProgress(
        title ? `正在处理《${title}》：${stageLabel(stage)}…` : "批量生成任务已在后端排队。",
        owner,
      );
      return;
    }

    // The queue response becomes terminal before the next state.json watcher
    // tick. Publish its chapter outcomes immediately so the current page
    // reflects completed audio without requiring a chapter reselect/reload.
    const terminalWorkflows = terminalGenerationWorkflowsFromBatch({
      chapters: response.chapters,
      updatedAt: response.updatedAt ?? Date.now() / 1000,
    });
    if (Object.keys(terminalWorkflows).length > 0) {
      setWorkflowStatuses(
        "generation",
        (previous) => ({ ...previous, ...terminalWorkflows }),
        owner,
      );
    }

    // A completed-with-errors batch may have produced usable audio for some
    // chapters, but it must not look like a clean 100% success in the UI.
    setProgress(
      total > 0 ? Math.round((completed / total) * 100) : 100,
      owner,
    );
    setProgressDetail([], owner);
    if (response.status === "succeeded") {
      setAnalyzeProgress(`批量生成完成：${response.succeededCount ?? total} 章已完成。`, owner);
      setSavedMessage("原章节配音、背景音/音效和最终混音已全部生成。", owner);
      setStage("done", owner);
      setCurrentStepRef.current("done");
    } else if (response.status === "completed_with_errors") {
      const failures = response.chapters
        .filter((chapter) => chapter.status === "failed")
        .slice(0, 2)
        .map((chapter) => `《${chapter.title}》：${chapter.error || "失败"}`)
        .join("；");
      setAnalyzeProgress(
        `批量生成完成：${response.succeededCount ?? 0} 章成功，${response.failedCount ?? 0} 章失败。`,
        owner,
      );
      setError(failures || "部分章节生成失败。", owner);
      setStage("done", owner);
    } else if (response.status === "cancelled") {
      setAnalyzeProgress(`任务已停止，已完成 ${response.succeededCount ?? 0} 章。`, owner);
      setStage("idle", owner);
    } else {
      setError(batchErrorMessage(response), owner);
      setAnalyzeProgress("批量生成失败。", owner);
      setStage("error", owner);
    }
  }, [
    setAnalyzeProgress,
    setAudioAssets,
    setChapterAudioPaths,
    setChapterMixedAudioPaths,
    setError,
    setGenerationBatch,
    setProgress,
    setProgressDetail,
    setSavedMessage,
    setStage,
    setWorkflowStatuses,
  ]);

  const pollBatch = useCallback(async (batchId: string, owner: string) => {
    if (requestInFlightRef.current) return;
    requestInFlightRef.current = true;
    try {
      const response = await getBatchGenerationStatus(batchId);
      if (activeBookIdRef.current !== owner) return;
      publishBatch(response, owner);
      if (isActiveBatchGeneration(response.status)) {
        pollTimerRef.current = window.setTimeout(() => {
          void pollBatch(batchId, owner);
        }, 1200);
      } else {
        activeBatchIdRef.current = null;
        stopPolling();
      }
    } catch (error) {
      if (activeBookIdRef.current === owner) {
        setError(`无法读取批量生成进度：${String(error)}`, owner);
        setStage("error", owner);
      }
      activeBatchIdRef.current = null;
      stopPolling();
    } finally {
      requestInFlightRef.current = false;
    }
  }, [publishBatch, setError, setStage, stopPolling]);

  const beginBatch = useCallback(async (chapterIds: string[], options: { force?: boolean; cacheSegments?: boolean } = {}) => {
    if (!book || !analysis || chapterIds.length === 0) {
      if (book) setError("请选择至少一个已分析章节。", book.bookId);
      return;
    }
    const validIds = chapterIds.filter((chapterId) => Boolean(analysis.scriptPaths[chapterId]));
    if (validIds.length === 0) {
      setError("所选章节尚未完成文本分析，无法开始全流程生成。", book.bookId);
      return;
    }
    stopPolling();
    setError(null, book.bookId);
    setSavedMessage(null, book.bookId);
    setStage("generating", book.bookId);
    setProgress(0, book.bookId);
    setAnalyzeProgress("正在提交后端批量生成队列…", book.bookId);
    try {
      const response = await startBatchGeneration({
        bookId: book.bookId,
        chapterIds: validIds,
        force: options.force === true,
        cacheSegments: options.cacheSegments !== false,
      });
      if (!response.batchId) throw new Error(batchErrorMessage(response));
      activeBatchIdRef.current = response.batchId;
      activeBookIdRef.current = book.bookId;
      publishBatch(response, book.bookId);
      if (isActiveBatchGeneration(response.status)) void pollBatch(response.batchId, book.bookId);
    } catch (error) {
      setError(`无法启动批量生成：${String(error)}`, book.bookId);
      setAnalyzeProgress("批量生成未启动。", book.bookId);
      setStage("error", book.bookId);
    }
  }, [
    analysis,
    book,
    pollBatch,
    publishBatch,
    setAnalyzeProgress,
    setError,
    setProgress,
    setSavedMessage,
    setStage,
    stopPolling,
  ]);

  useEffect(() => {
    activeBookIdRef.current = book?.bookId ?? null;
    if (!book) return;
    let cancelled = false;
    void getActiveBatchGeneration(book.bookId)
      .then((response) => {
        if (cancelled || !response || !response.batchId || !isActiveBatchGeneration(response.status)) return;
        activeBatchIdRef.current = response.batchId;
        publishBatch(response, book.bookId);
        void pollBatch(response.batchId, book.bookId);
      })
      .catch(() => {
        // Opening a book must remain possible if an older backend is offline.
      });
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [book?.bookId, pollBatch, publishBatch, stopPolling]);

  const handleGenerate = useCallback(async () => {
    if (!book || !analysis) return;
    // A Set keeps checkbox insertion order, which can differ from the book's
    // chapter order when people select chapters out of order. The backend queue
    // must always follow the book timeline.
    const selected = book.chapters
      .map((chapter) => chapter.id)
      .filter((chapterId) => selectedChapters.has(chapterId) && analysis.scriptPaths[chapterId]);
    const chapterIds = correctionState.affectedChapters.length > 0
      ? selected.filter((chapterId) => correctionState.affectedChapters.includes(chapterId))
      : selected;
    await beginBatch(chapterIds);
  }, [analysis, beginBatch, book, correctionState.affectedChapters, selectedChapters]);

  const handleStopGeneration = useCallback(async () => {
    const batchId = activeBatchIdRef.current;
    if (!book || !batchId) return;
    try {
      const response = await cancelBatchGeneration(batchId);
      publishBatch(response, book.bookId);
      setAnalyzeProgress("已请求停止；正在完成当前 worker 阶段后停止后续章节。", book.bookId);
    } catch (error) {
      setError(`停止批量生成失败：${String(error)}`, book.bookId);
    }
  }, [book, publishBatch, setAnalyzeProgress, setError]);

  const handleRegenerateChapter = useCallback(async (chapter: ChapterMeta) => {
    await beginBatch([chapter.id], { force: true, cacheSegments: false });
  }, [beginBatch]);

  const handleRegenerateAll = useCallback(async () => {
    if (!book) return;
    const chapterIds = book.chapters
      .filter((chapter) => chapterAudioPaths[chapter.id])
      .map((chapter) => chapter.id);
    await beginBatch(chapterIds, { force: true, cacheSegments: false });
  }, [beginBatch, book, chapterAudioPaths]);

  const handleRegenerateAudioAsset = useCallback(async (asset: AudioAsset, chapter: ChapterMeta) => {
    if (!book || !analysis?.scriptPaths[chapter.id]) return;
    setStage("generating", book.bookId);
    setError(null, book.bookId);
    setAnalyzeProgress(`正在本地重新生成 ${asset.kind === "music" ? "背景音乐" : "音效"} ${asset.assetId}…`, book.bookId);
    try {
      const result = await workerCall("generate_audio_assets", {
        bookId: book.bookId,
        chapterId: chapter.id,
        scriptPath: analysis.scriptPaths[chapter.id],
        outputDirectory: `${book.workDir}/audio-assets/${chapter.id}`,
        mixedOutputPath: `${book.workDir}/audio/${chapter.id}_mixed.wav`,
        force: true,
        assetId: asset.assetId,
        assetKind: asset.kind,
      });
      if (result.status !== "succeeded") {
        throw new Error((result.error as { message?: string })?.message || "背景音/音效生成失败。");
      }
      const assets = audioAssetsFromArtifacts(result.artifacts).map((item) => ({
        ...item,
        refreshKey: `${Date.now()}-${item.kind}-${item.assetId}`,
      }));
      setAudioAssets((previous) => ({ ...previous, [chapter.id]: assets }), book.bookId);
      setChapterMixedAudioPaths((previous) => {
        const next = { ...previous };
        delete next[chapter.id];
        return next;
      }, book.bookId);
      setSavedMessage("背景音/音效已重新生成；可选择该章节重新执行全流程生成最终混音。", book.bookId);
      setStage("idle", book.bookId);
    } catch (error) {
      setError(`背景音/音效生成失败：${String(error)}`, book.bookId);
      setStage("error", book.bookId);
    }
  }, [analysis, book, setAnalyzeProgress, setAudioAssets, setChapterMixedAudioPaths, setError, setSavedMessage, setStage]);

  return {
    handleGenerate,
    handleStopGeneration,
    handleRegenerateAudioAsset,
    handleRegenerateChapter,
    handleRegenerateAll,
  };
}
