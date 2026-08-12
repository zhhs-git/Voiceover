import { useRef, useCallback, useEffect, useMemo, useState } from "react";
import { convertFileSrc, downloadFile, invoke } from "../lib/platform";
import type {
  AnalysisState,
  AudioAsset,
  BookState,
  ChapterMeta,
  CharacterMeta,
  LibraryBook,
  VoiceMeta,
  VoiceOption,
} from "../types";
import { createAudiobookStore } from "../state/store";
import { useCorrectionStore } from "../state/corrections";
import { usePipelineStore } from "../state/pipelineStore";
import { useChapterAnalysis } from "../hooks/useChapterAnalysis";
import { useGeneration } from "../hooks/useGeneration";
import { workerCall } from "../lib/workerCall";
import {
  audioAssetsFromManifest,
  filterAudioAssetsToScriptPlan,
} from "../lib/audioAssets";
import { buildVoiceOptions, localizeVoiceDisplayName } from "../lib/voiceOptions";
import { ChapterPreview } from "./ChapterPreview";
import { GeneratedAudioAssetList } from "./GeneratedAudioAssetList";
import { PersistentChapterAudioList } from "./PersistentChapterAudioList";
import { BatchGenerationStatusList } from "./BatchGenerationStatusList";
import {
  buildVoicePreviewRequest,
  buildVoicePreviewScript,
} from "../lib/voicePreview";
import {
  emptyChapterWorkflow,
  readChapterWorkflow,
  watchChapterWorkflow,
} from "../lib/workflowStatus";
import { WorkflowSteps, workflowStatusLabel } from "./WorkflowSteps";

const db = createAudiobookStore();

const AGE_LABELS: Record<string, string> = {
  child: "儿童",
  young: "青年",
  adult: "成年",
  older: "年长",
  unknown: "未知",
};

function ageLabel(value?: string | null): string {
  return AGE_LABELS[value ?? "unknown"] ?? value ?? "未知";
}

// Extract the last meaningful line from a Python traceback / worker stderr dump.
function cleanWorkerError(raw: string): string {
  const body = raw.replace(/^Worker exited[^:]*:\s*/s, "");
  const lines = body.split("\n").map((l) => l.trim()).filter(Boolean);
  return lines.at(-1) ?? raw;
}

function parseAliases(value: string | null | undefined): string[] {
  try {
    const parsed = JSON.parse(value || "[]");
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function normalizeCharacter(value: Partial<CharacterMeta>): CharacterMeta | null {
  if (typeof value.id !== "string" || typeof value.canonicalName !== "string") {
    return null;
  }
  return {
    id: value.id,
    canonicalName: value.canonicalName,
    aliases: Array.isArray(value.aliases)
      ? value.aliases.filter((alias): alias is string => typeof alias === "string")
      : [],
    gender: typeof value.gender === "string" ? value.gender : "unknown",
    ageClass: typeof value.ageClass === "string" ? value.ageClass : "unknown",
    identityStatus:
      value.identityStatus === "provisional" ||
      value.identityStatus === "confirmed" ||
      value.identityStatus === "merged"
        ? value.identityStatus
        : undefined,
    voiceId: typeof value.voiceId === "string" ? value.voiceId : "neutral_dialogue_01",
    voiceSource:
      value.voiceSource === "auto" || value.voiceSource === "manual"
        ? value.voiceSource
        : undefined,
    voiceAssignmentVersion:
      typeof value.voiceAssignmentVersion === "number"
        ? value.voiceAssignmentVersion
        : undefined,
    voiceProfile: typeof value.voiceProfile === "string" ? value.voiceProfile : undefined,
    fallbackVoiceId:
      typeof value.fallbackVoiceId === "string" ? value.fallbackVoiceId : undefined,
    voiceDesign:
      typeof value.voiceDesign === "string" ? value.voiceDesign : undefined,
    voiceDescription:
      typeof value.voiceDescription === "string" ? value.voiceDescription : undefined,
    confidence: typeof value.confidence === "number" ? value.confidence : 0.5,
  };
}

const GENERIC_CHARACTER_LABELS = new Set([
  "小姐", "少爷", "姑娘", "公子", "夫人", "太太", "老爷", "殿下", "陛下",
  "皇上", "皇后", "公主", "王爷", "世子", "大人", "先生", "女士", "母亲",
  "父亲", "娘", "爹", "妈妈", "爸爸", "mother", "father", "wife", "husband",
  "miss", "mrs", "ms", "mr", "sir", "madam", "lady", "lord", "girl", "boy",
  "woman", "man",
]);

function characterKey(value: string): string {
  return value.trim().toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
}

function characterKeys(character: { id: string; canonicalName: string; aliases: string[] }): string[] {
  const id = characterKey(character.id);
  const names = [character.canonicalName, ...character.aliases]
    .map(characterKey)
    .filter((value) => value && !GENERIC_CHARACTER_LABELS.has(value));
  return [...new Set([id, ...names].filter(Boolean))];
}

function applyVoiceAssignment(target: CharacterMeta, source: CharacterMeta) {
  target.voiceId = source.voiceId;
  target.voiceSource = source.voiceSource;
  target.voiceAssignmentVersion = source.voiceAssignmentVersion;
  target.voiceProfile = source.voiceProfile;
  target.fallbackVoiceId = source.fallbackVoiceId;
  target.voiceDesign = source.voiceDesign;
  target.voiceDescription = source.voiceDescription;
}

function assignmentVersion(character: CharacterMeta): number {
  return character.voiceAssignmentVersion ?? 0;
}

function mergeCharacters(characters: CharacterMeta[]): CharacterMeta[] {
  const merged: CharacterMeta[] = [];
  const keyToIndex = new Map<string, number>();
  for (const character of characters) {
    const keys = characterKeys(character);
    const existingIndex = keys.map((key) => keyToIndex.get(key)).find((index) => index !== undefined);
    if (existingIndex === undefined) {
      const copy = { ...character, aliases: [...new Set(character.aliases)] };
      merged.push(copy);
      for (const key of keys) keyToIndex.set(key, merged.length - 1);
      continue;
    }
    const existing = merged[existingIndex];
    existing.aliases = [...new Set([...existing.aliases, ...character.aliases, character.canonicalName])]
      .filter((alias) => alias !== existing.canonicalName);
    if (character.confidence > existing.confidence) existing.confidence = character.confidence;
    if (existing.gender === "unknown" && character.gender !== "unknown") existing.gender = character.gender;
    if ((existing.ageClass ?? "unknown") === "unknown" && (character.ageClass ?? "unknown") !== "unknown") {
      existing.ageClass = character.ageClass;
    }
    if (character.identityStatus === "confirmed" || !existing.identityStatus) {
      existing.identityStatus = character.identityStatus;
    }
    const existingManual = existing.voiceSource === "manual";
    const incomingManual = character.voiceSource === "manual";
    // Script records are the source of truth for automatic routing.  A manual
    // selection remains authoritative even when an old DB row is restored
    // before a newer script is read.
    if (
      incomingManual ||
      (!existingManual && Boolean(character.voiceId) && assignmentVersion(character) >= assignmentVersion(existing)) ||
      (!existing.voiceId || existing.voiceId === "neutral_dialogue_01")
    ) {
      applyVoiceAssignment(existing, character);
    }
    for (const key of keys) keyToIndex.set(key, existingIndex);
  }
  return merged;
}

interface BookDetailViewProps {
  libraryBook: LibraryBook;
  book: BookState;
  sourcePath: string;
  onBack: () => void;
  onBookUpdate: (book: BookState) => void;
}

export function BookDetailView({
  libraryBook,
  book,
  sourcePath,
  onBack,
  onBookUpdate,
}: BookDetailViewProps) {
  const pipeline = usePipelineStore();
  const correctionState = useCorrectionStore();
  const narratorVoiceId = book.narratorVoiceId ?? "narrator_female";
  const abortRef = useRef<AbortController | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const [providerVoices, setProviderVoices] = useState<VoiceMeta[]>([]);

  const voiceOptions: VoiceOption[] = useMemo(
    () =>
      buildVoiceOptions({
        providerVoices,
        analysisVoices: pipeline.analysis?.voices ?? [],
        characters: pipeline.analysis?.characters ?? [],
      }),
    [providerVoices, pipeline.analysis?.voices, pipeline.analysis?.characters],
  );

  const isBusy =
    pipeline.stage === "importing" ||
    pipeline.stage === "analyzing" ||
    pipeline.stage === "saving" ||
    pipeline.stage === "generating";
  const selectedChapterId =
    pipeline.selectedChapters.size === 1
      ? [...pipeline.selectedChapters][0]
      : undefined;
  const runningBatchChapterId = pipeline.generationBatch?.chapters.find(
    (chapter) => chapter.status === "running",
  )?.chapterId;
  const activeGenerationChapterId = runningBatchChapterId ?? selectedChapterId;
  const activeGenerationWorkflow = activeGenerationChapterId
    ? pipeline.workflows.generation[activeGenerationChapterId]
    : undefined;
  const activeGenerationTitle = activeGenerationChapterId
    ? book.chapters.find((chapter) => chapter.id === activeGenerationChapterId)?.title
    : undefined;
  const [listenChapterId, setListenChapterId] = useState<string>();

  const listenableChapters = useMemo(() => {
    const chapterIds = new Set<string>([
      ...Object.keys(pipeline.chapterAudioPaths),
      ...Object.keys(pipeline.chapterMixedAudioPaths),
      ...Object.entries(pipeline.audioAssets)
        .filter(([, assets]) => assets.length > 0)
        .map(([chapterId]) => chapterId),
    ]);
    return book.chapters.filter((chapter) => chapterIds.has(chapter.id));
  }, [
    book.chapters,
    pipeline.audioAssets,
    pipeline.chapterAudioPaths,
    pipeline.chapterMixedAudioPaths,
  ]);

  // The chapter checkboxes drive batch operations.  Listening is intentionally
  // independent so selecting several chapters does not make the preview list
  // expand back to the whole book.
  useEffect(() => {
    setListenChapterId((current) => {
      if (current && listenableChapters.some((chapter) => chapter.id === current)) {
        return current;
      }
      if (selectedChapterId && listenableChapters.some((chapter) => chapter.id === selectedChapterId)) {
        return selectedChapterId;
      }
      return listenableChapters[0]?.id;
    });
  }, [listenableChapters, selectedChapterId]);

  const activeListenChapterId =
    listenChapterId && listenableChapters.some((chapter) => chapter.id === listenChapterId)
      ? listenChapterId
      : selectedChapterId && listenableChapters.some((chapter) => chapter.id === selectedChapterId)
        ? selectedChapterId
        : listenableChapters[0]?.id;

  const noopSetCurrentStep = () => {};

  const { handleAnalyze } = useChapterAnalysis({
    book,
    analysis: pipeline.analysis,
    selectedChapters: pipeline.selectedChapters,
    setStage: pipeline.setStage,
    setError: pipeline.setError,
    setSavedMessage: pipeline.setSavedMessage,
    setAnalyzeProgress: pipeline.setAnalyzeProgress,
    setChapterStatuses: pipeline.setChapterStatuses,
    setProgressDetail: pipeline.setProgressDetail,
    setProgress: pipeline.setProgress,
    setWorkflowStatus: pipeline.setWorkflowStatus,
    setAnalysis: pipeline.setAnalysis,
    setChapterAudioPaths: pipeline.setChapterAudioPaths,
    setChapterMixedAudioPaths: pipeline.setChapterMixedAudioPaths,
    setAudioAssets: pipeline.setAudioAssets,
    setCurrentStep: noopSetCurrentStep,
    setTab: pipeline.setTab,
    abortRef,
    db,
  });

  const {
    handleGenerate,
    handleStopGeneration,
    handleRegenerateAudioAsset,
    handleRegenerateChapter,
    handleRegenerateAll,
  } =
    useGeneration({
      book,
      analysis: pipeline.analysis,
      selectedChapters: pipeline.selectedChapters,
      chapterAudioPaths: pipeline.chapterAudioPaths,
      correctionState: correctionState as {
        affectedChapters: string[];
        dirty?: boolean;
      },
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
      setCurrentStep: noopSetCurrentStep,
      abortRef,
    });

  useEffect(() => {
    return () => {
      // A generation batch is owned by the backend. Leaving or refreshing the
      // web page must only stop local polling, never cancel the batch.
      abortRef.current?.abort();
    };
  }, [book.bookId]);

  // Keep selected chapters synchronized after a page refresh as well as while
  // a long-running worker is still active.  The hook-level watchers provide
  // immediate updates during button actions; this watcher is the durable
  // recovery path for an already-running or needs-review workflow.
  useEffect(() => {
    const stops: Array<() => void> = [];
    for (const chapterId of pipeline.selectedChapters) {
      for (const kind of ["analysis", "generation"] as const) {
        stops.push(
          watchChapterWorkflow(
            book.workDir,
            chapterId,
            kind,
            (workflow) => pipeline.setWorkflowStatus(kind, chapterId, workflow, book.bookId),
            1500,
          ),
        );
      }
    }
    return () => stops.forEach((stop) => stop());
  }, [book.bookId, book.workDir, pipeline.selectedChapters, pipeline.setWorkflowStatus]);

  // Restore saved state on mount
  useEffect(() => {
    let cancelled = false;
    pipeline.activateBook(book.bookId);
    async function restore() {
      // Load characters
      const chars = await db.getCharacters(book.bookId).catch(() => []);
      if (cancelled) return;

      const restoredCharacters: CharacterMeta[] = chars.map(
        (c): CharacterMeta => ({
          id: c.id,
          canonicalName: c.canonicalName,
          aliases: parseAliases(c.aliases),
          gender: c.gender || "unknown",
          ageClass: c.ageClass || "unknown",
          identityStatus:
            c.identityStatus === "provisional" ||
            c.identityStatus === "confirmed" ||
            c.identityStatus === "merged"
              ? c.identityStatus
              : undefined,
          voiceId: c.voiceId || "narrator_default",
          voiceSource:
            c.voiceSource === "auto" || c.voiceSource === "manual"
              ? c.voiceSource
              : undefined,
          voiceAssignmentVersion:
            typeof c.voiceAssignmentVersion === "number"
              ? c.voiceAssignmentVersion
              : undefined,
          voiceProfile: c.voiceProfile || undefined,
          fallbackVoiceId: c.fallbackVoiceId || undefined,
          voiceDesign: c.voiceDesign || undefined,
          voiceDescription: c.voiceDescription || undefined,
          confidence: c.confidence,
        }),
      );

      // Load scripts (analyzed chapters)
      const chaptersWithScripts = await db
        .getChaptersWithScripts(book.bookId)
        .catch(() => []);
      if (cancelled) return;
      const scriptPaths: Record<string, string> = {};
      const scriptCharacters: CharacterMeta[] = [];
      const restoredVoices: VoiceMeta[] = [];
      const scriptDataByChapter: Record<string, unknown> = {};
      const restoredAudioAssets: Record<string, AudioAsset[]> = {};
      for (const ch of chaptersWithScripts) {
        scriptPaths[ch.id] = ch.scriptPath;
        try {
          const scriptRaw = await invoke<string>("run_worker", {
            command: "_read_file",
            inputJson: JSON.stringify({ path: ch.scriptPath }),
          });
          const scriptData = JSON.parse(scriptRaw) as {
            characters?: Array<Partial<CharacterMeta>>;
            voices?: VoiceMeta[];
            audioPlan?: unknown;
          } | null;
          scriptDataByChapter[ch.id] = scriptData;
          for (const character of scriptData?.characters ?? []) {
            const normalized = normalizeCharacter(character);
            if (normalized) scriptCharacters.push(normalized);
          }
          for (const voice of scriptData?.voices ?? []) {
            if (
              typeof voice?.id === "string" &&
              !restoredVoices.some((existing) => existing.id === voice.id)
            ) {
              restoredVoices.push(voice);
            }
          }
        } catch {
          // A missing or malformed script should not prevent the book from opening.
        }
      }

      await Promise.all(
        book.chapters.map(async (chapter) => {
          const manifestPath = `${book.workDir}/audio-assets/${chapter.id}/manifest.json`;
          try {
            const manifestRaw = await invoke<string>("run_worker", {
              command: "_read_file",
              inputJson: JSON.stringify({ path: manifestPath }),
            });
            const assets = filterAudioAssetsToScriptPlan(
              audioAssetsFromManifest(JSON.parse(manifestRaw)),
              scriptDataByChapter[chapter.id],
            );
            if (assets.length > 0) restoredAudioAssets[chapter.id] = assets;
          } catch {
            // Audio assets are optional; a missing manifest is expected.
          }
        }),
      );

      if (cancelled) return;
      // Restore one complete, book-scoped snapshot. Never merge with the
      // global pipeline's previous book state.
      const restoredRoster = mergeCharacters([
        ...restoredCharacters,
        ...scriptCharacters,
      ]);
      // The database roster is the book-level identity registry. Keep entries
      // even when a character has not yet been referenced by a loaded script;
      // otherwise a later chapter could analyze against an incomplete roster.
      const characters = restoredRoster;
      pipeline.setAnalysis(
        {
          characters,
          voices: restoredVoices,
          scriptPaths,
        },
        book.bookId,
      );
      pipeline.setAudioAssets(restoredAudioAssets, book.bookId);

      const restoredAnalysisWorkflows: Record<string, ReturnType<typeof emptyChapterWorkflow>> = {};
      const restoredGenerationWorkflows: Record<string, ReturnType<typeof emptyChapterWorkflow>> = {};
      await Promise.all(
        book.chapters.map(async (chapter) => {
          const [analysisWorkflow, generationWorkflow] = await Promise.all([
            readChapterWorkflow(book.workDir, chapter.id, "analysis"),
            readChapterWorkflow(book.workDir, chapter.id, "generation"),
          ]);
          restoredAnalysisWorkflows[chapter.id] = analysisWorkflow;
          restoredGenerationWorkflows[chapter.id] = generationWorkflow;
        }),
      );
      if (cancelled) return;
      pipeline.setWorkflowStatuses("analysis", restoredAnalysisWorkflows, book.bookId);
      pipeline.setWorkflowStatuses("generation", restoredGenerationWorkflows, book.bookId);

      // Restore the original voice track and the final mixed track separately.
      const audioCandidates = book.chapters.flatMap((ch) => [
        `${book.workDir}/audio/${ch.id}_mixed.wav`,
        `${book.workDir}/audio/${ch.id}.wav`,
      ]);
      const existing: string[] = await invoke("file_exists", { paths: audioCandidates });
      if (cancelled) return;
      const existingSet = new Set(existing);
      const voicePaths: Record<string, string> = {};
      const mixedPaths: Record<string, string> = {};
      for (const chapter of book.chapters) {
        const mixedPath = `${book.workDir}/audio/${chapter.id}_mixed.wav`;
        const voicePath = `${book.workDir}/audio/${chapter.id}.wav`;
        if (existingSet.has(mixedPath)) {
          mixedPaths[chapter.id] = mixedPath;
        }
        if (existingSet.has(voicePath)) {
          voicePaths[chapter.id] = voicePath;
        }
      }
      pipeline.setChapterAudioPaths(voicePaths, book.bookId);
      pipeline.setChapterMixedAudioPaths(mixedPaths, book.bookId);
      // Auto-select chapters that need analysis (no script yet)
      const unanalyzed = book.chapters.filter((c) => !scriptPaths[c.id]).map((c) => c.id);
      pipeline.setSelectedChapters(new Set(unanalyzed), book.bookId);
    }
    restore();
    return () => {
      cancelled = true;
    };
  }, [book.bookId, book.workDir, book.chapters]);

  useEffect(() => {
    let cancelled = false;
    workerCall("list_voices", { backend: "mimo" })
      .then((result) => {
        if (!cancelled && Array.isArray(result.voices)) {
          setProviderVoices(result.voices as VoiceMeta[]);
        }
      })
      .catch(() => {
        // Script voices and current assignments remain the fallback catalog.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function toggleChapter(chapterId: string) {
    pipeline.setSelectedChapters((prev) => {
      const next = new Set(prev);
      next.has(chapterId) ? next.delete(chapterId) : next.add(chapterId);
      return next;
    }, book.bookId);
  }

  const allSelected = useMemo(
    () =>
      book.chapters.length > 0 &&
      book.chapters.every((c) => pipeline.selectedChapters.has(c.id)),
    [book.chapters, pipeline.selectedChapters],
  );

  function toggleAllChapters() {
    pipeline.setSelectedChapters(
      allSelected
        ? new Set()
        : new Set(book.chapters.map((c) => c.id)),
      book.bookId,
    );
  }

  async function handleReextract() {
    pipeline.setStage("importing", book.bookId);
    pipeline.setAnalyzeProgress("正在从数据库恢复章节…", book.bookId);
    try {
      const rows = await db.getChapters(book.bookId);
      if (usePipelineStore.getState().bookId !== book.bookId) return;

      const existingScripts = pipeline.analysis?.scriptPaths ?? {};
      // Reconstruct ChapterMeta from DB rows; textPath follows the fixed layout
      // written by the extract_book worker: workDir/chapters/{id}.txt
      const freshChapters = rows.map((r) => ({
        id: r.id,
        title: r.title,
        textLength: 0,
        textPath: `${book.workDir}/chapters/${r.id}.txt`,
      }));

      if (freshChapters.length === 0) {
        pipeline.setError("数据库中没有章节记录，请重新导入书籍。", book.bookId);
        pipeline.setStage("error", book.bookId);
        return;
      }

      // Auto-select chapters that don't have scripts yet
      const unanalyzed = freshChapters
        .filter((c) => !existingScripts[c.id])
        .map((c) => c.id);
      pipeline.setSelectedChapters(new Set(unanalyzed), book.bookId);
      pipeline.setTab("analyze", book.bookId);

      // Update the book state in App so the sidebar re-renders with fresh chapters
      onBookUpdate({ ...book, chapters: freshChapters });

      const analyzedCount = Object.keys(existingScripts).length;
      pipeline.setAnalyzeProgress(
        `找到 ${freshChapters.length} 章` +
        (analyzedCount > 0 ? ` · ${analyzedCount} 章已分析` : "") +
        ` · 已选择 ${unanalyzed.length} 章待分析`,
        book.bookId,
      );
      pipeline.setStage("idle", book.bookId);
    } catch (err) {
      pipeline.setError(`重新提取失败：${String(err)}`, book.bookId);
      pipeline.setStage("error", book.bookId);
    }
  }

  async function handleDownloadChapter(
    chapterId: string,
    chapterTitle: string,
    audioPathOverride?: string,
  ) {
    const wavPath = audioPathOverride ?? pipeline.chapterAudioPaths[chapterId];
    if (!wavPath) return;
    const safeName = chapterTitle.replace(/[/\\:*?"<>|]/g, "_");
    const suffix = audioPathOverride ? "_最终配音" : "";
    const filename = `${safeName}${suffix}.mp3`;
    const dest = `${book.workDir}/downloads/${filename}`;
    pipeline.setSavedMessage(`正在转换"${chapterTitle}"…`, book.bookId);
    try {
      const result = await workerCall("convert_to_mp3", {
        wavPath,
        outputPath: dest,
      });
      if (result.status !== "succeeded") {
        throw new Error((result.error as any)?.message ?? "转换失败");
      }
      downloadFile(convertFileSrc(dest), filename);
      pipeline.setSavedMessage(`"${chapterTitle}"已保存为 MP3。`, book.bookId);
    } catch (err) {
      pipeline.setError(`下载失败：${String(err)}`, book.bookId);
    }
  }

  async function handleDownloadAudioAsset(asset: AudioAsset, chapter: ChapterMeta) {
    const chapterName = chapter.title.replace(/[/\\:*?"<>|]/g, "_");
    const assetName = (asset.path.split("/").pop() || asset.assetId)
      .replace(/\.[^.]+$/, "")
      .replace(/[/\\:*?"<>|]/g, "_");
    const kindName = asset.kind === "music" ? "music" : "sfx";
    const filename = `${chapterName}_${kindName}_${assetName}.mp3`;
    const destination = `${book.workDir}/downloads/${filename}`;
    pipeline.setSavedMessage(`正在转换“${assetName}”为 MP3…`, book.bookId);
    try {
      const result = await workerCall("convert_to_mp3", {
        wavPath: asset.path,
        outputPath: destination,
      });
      if (result.status !== "succeeded") {
        throw new Error((result.error as { message?: string } | undefined)?.message ?? "转换失败");
      }
      downloadFile(convertFileSrc(destination), filename);
      pipeline.setSavedMessage(`已下载 MP3：${filename}`, book.bookId);
    } catch (error) {
      pipeline.setError(`背景音/音效下载失败：${String(error)}`, book.bookId);
    }
  }

  function handleStop() {
    abortRef.current?.abort();
    if (pipeline.stage === "generating") {
      void handleStopGeneration();
      return;
    }
    pipeline.setAnalyzeProgress("正在停止…", book.bookId);
  }

  const handleGenderChange = useCallback(
    (characterId: string, gender: string) => {
      correctionState.setGender(characterId, gender);
      pipeline.setSavedMessage(null, book.bookId);
    },
    [book.bookId, correctionState.setGender, pipeline.setSavedMessage],
  );

  const handleVoiceChange = useCallback(
    (characterId: string, voiceId: string) => {
      correctionState.setVoice(characterId, voiceId);
      pipeline.setAnalysis((current) => {
        if (!current) return current;
        return {
          ...current,
          characters: current.characters.map((c) =>
            c.id === characterId
              ? {
                  ...c,
                  voiceId,
                  voiceSource: "manual",
                  voiceAssignmentVersion: undefined,
                  voiceProfile: undefined,
                  fallbackVoiceId: undefined,
                  voiceDesign: undefined,
                  voiceDescription: undefined,
                }
              : c,
          ),
        };
      }, book.bookId);
      pipeline.setSavedMessage(null, book.bookId);
    },
    [
      book.bookId,
      correctionState.setVoice,
      pipeline.setAnalysis,
      pipeline.setSavedMessage,
    ],
  );

  const handleVoiceDesignChange = useCallback(
    (characterId: string, voiceDesign: string) => {
      correctionState.setVoiceDesign(characterId, voiceDesign);
      pipeline.setAnalysis((current) => {
        if (!current) return current;
        return {
          ...current,
          characters: current.characters.map((c) =>
            c.id === characterId
              ? {
                  ...c,
                  voiceDesign,
                  voiceDescription: voiceDesign,
                }
              : c,
          ),
        };
      }, book.bookId);
      pipeline.setSavedMessage(null, book.bookId);
    },
    [
      book.bookId,
      correctionState.setVoiceDesign,
      pipeline.setAnalysis,
      pipeline.setSavedMessage,
    ],
  );

  async function handleSaveCorrections() {
    if (!book || !pipeline.analysis) return;
    pipeline.setStage("saving", book.bookId);
    pipeline.setError(null, book.bookId);
    pipeline.setSavedMessage(null, book.bookId);
    try {
      const result = await workerCall("apply_corrections", {
        bookId: book.bookId,
        narratorVoiceId,
        chapters: book.chapters.map((c) => ({
          chapterId: c.id,
          textPath: c.textPath,
          title: c.title,
        })),
        corrections: {
          aliasMerges: correctionState.aliasMerges,
          genderOverrides: correctionState.genderOverrides,
          voiceOverrides: correctionState.voiceOverrides,
          voiceDesignOverrides: correctionState.voiceDesignOverrides,
        },
        // Keep the full roster available so correction-time re-analysis
        // reuses stable character IDs and does not rediscover duplicates.
        knownCharacters: pipeline.analysis.characters.map((character) => ({
          id: character.id,
          canonicalName: character.canonicalName,
          aliases: character.aliases,
          gender: character.gender,
          ageClass: character.ageClass,
          identityStatus: character.identityStatus,
          voiceId: character.voiceId,
          voiceSource: character.voiceSource,
          voiceAssignmentVersion: character.voiceAssignmentVersion,
          voiceProfile: character.voiceProfile,
          fallbackVoiceId: character.fallbackVoiceId,
          voiceDesign: character.voiceDesign,
          voiceDescription: character.voiceDescription,
        })),
        outputDirectory: `${book.workDir}/scripts`,
      });
      if (result.status !== "succeeded")
        throw new Error(
          (result.error as any)?.message ?? "应用修改失败",
        );
      if (usePipelineStore.getState().bookId !== book.bookId) return;
      const artifacts = result.artifacts as unknown as Array<{
        path: string;
        metadata: { chapterId: string };
      }>;
      const newScriptPaths = {
        ...pipeline.analysis.scriptPaths,
      };
      const affectedIds: string[] = [];
      for (const art of artifacts) {
        newScriptPaths[art.metadata.chapterId] = art.path;
        affectedIds.push(art.metadata.chapterId);
      }
      if (artifacts.length > 0) {
        const firstScriptRaw = await invoke<string>("run_worker", {
          command: "_read_file",
          inputJson: JSON.stringify({ path: artifacts[0].path }),
        }).catch(() => "{}");
        if (usePipelineStore.getState().bookId !== book.bookId) return;
        const firstScript = JSON.parse(firstScriptRaw) as {
          characters?: CharacterMeta[];
          voices?: VoiceMeta[];
        } | null;
        if (firstScript?.characters) {
          const correctedCharacters = firstScript.characters;
          await Promise.all(
            correctedCharacters.map((character) =>
              db.upsertCharacter({
                id: character.id,
                bookId: book.bookId,
                canonicalName: character.canonicalName,
                gender: character.gender,
                ageClass: character.ageClass,
                identityStatus: character.identityStatus,
                voiceId: character.voiceId,
                voiceSource: character.voiceSource,
                voiceAssignmentVersion: character.voiceAssignmentVersion,
                voiceProfile: character.voiceProfile,
                fallbackVoiceId: character.fallbackVoiceId,
                voiceDesign: character.voiceDesign,
                voiceDescription: character.voiceDescription,
                confidence: character.confidence,
                aliases: JSON.stringify(character.aliases),
              }),
            ),
          );
          pipeline.setAnalysis((current) => {
            if (!current) return current;
            return {
              ...current,
              scriptPaths: newScriptPaths,
              characters: mergeCharacters([
                ...current.characters,
                ...correctedCharacters,
              ]),
            };
          }, book.bookId);
        } else {
          pipeline.setAnalysis((current) =>
            current ? { ...current, scriptPaths: newScriptPaths } : current,
          book.bookId);
        }
      }
      correctionState.markSaved(affectedIds);
      if (affectedIds.length > 0) {
        const affected = new Set(affectedIds);
        pipeline.setChapterAudioPaths((prev) => {
          const next = { ...prev };
          for (const chapterId of affected) delete next[chapterId];
          return next;
        }, book.bookId);
        pipeline.setChapterMixedAudioPaths((prev) => {
          const next = { ...prev };
          for (const chapterId of affected) delete next[chapterId];
          return next;
        }, book.bookId);
        pipeline.setAudioAssets((prev) => {
          const next = { ...prev };
          for (const chapterId of affected) delete next[chapterId];
          return next;
        }, book.bookId);
      }
      pipeline.setSavedMessage(
        `${affectedIds.length} 个章节已更新。`,
        book.bookId,
      );
      pipeline.setStage("idle", book.bookId);
    } catch (err) {
      pipeline.setError(`保存修改失败：${String(err)}`, book.bookId);
      pipeline.setStage("error", book.bookId);
    }
  }

  async function handleNarratorVoiceChange(
    nextVoiceId: "narrator_female" | "narrator_male",
  ) {
    if (!book || nextVoiceId === narratorVoiceId) return;
    try {
      await db.setNarratorVoice(book.bookId, nextVoiceId);
      onBookUpdate({ ...book, narratorVoiceId: nextVoiceId });
      pipeline.setChapterAudioPaths((previous) => {
        const next = { ...previous };
        for (const chapter of book.chapters) delete next[chapter.id];
        return next;
      }, book.bookId);
      pipeline.setChapterMixedAudioPaths((previous) => {
        const next = { ...previous };
        for (const chapter of book.chapters) delete next[chapter.id];
        return next;
      }, book.bookId);
      pipeline.setSavedMessage(
        `已固定为${nextVoiceId === "narrator_male" ? "男性" : "女性"}旁白，旧旁白音频需要重新生成。`,
        book.bookId,
      );
    } catch (error) {
      pipeline.setError(`保存旁白音色失败：${String(error)}`, book.bookId);
      pipeline.setStage("error", book.bookId);
    }
  }

  async function handlePreviewVoice(
    voiceId: string,
    voiceDescription?: string,
    fallbackVoiceId?: string,
  ) {
    if (!book) return;
    try {
      pipeline.setSavedMessage(
        `正在生成 ${voiceId} 的试听音频…`,
        book.bookId,
      );
      const previewDir = `${book.workDir}/voice-previews`;
      const scriptPath = `${previewDir}/${voiceId}.json`;
      await workerCall("_write_file", {
        path: scriptPath,
        content: JSON.stringify(buildVoicePreviewScript(book.bookId, voiceId, {
          voiceDescription,
          fallbackVoiceId,
        })),
      });
      const segmentId = `preview_${voiceId}`;
      const result = await workerCall(
        "synthesize_segment_audio",
        buildVoicePreviewRequest(
          scriptPath,
          previewDir,
          segmentId,
          `${book.workDir}/voice-profiles`,
        ),
      );
      if (result.status !== "succeeded")
        throw new Error(
          (result.error as any)?.message ?? "音色试听生成失败",
        );
      const audioPath = (result.artifacts as Array<{ path: string }>)[0]?.path;
      if (!audioPath) throw new Error("音色试听未返回音频文件");
      const existingPaths = await invoke<string[]>("file_exists", {
        paths: [audioPath],
      });
      if (!existingPaths.includes(audioPath)) {
        throw new Error("音色试听音频文件未生成");
      }
      previewAudioRef.current?.pause();
      const audio = new Audio(convertFileSrc(audioPath));
      previewAudioRef.current = audio;
      audio.addEventListener(
        "ended",
        () => {
          if (previewAudioRef.current === audio) previewAudioRef.current = null;
        },
        { once: true },
      );
      await audio.play();
      if (usePipelineStore.getState().bookId !== book.bookId) return;
      pipeline.setSavedMessage(`正在播放 ${voiceId} 的试听音频。`, book.bookId);
    } catch (err) {
      pipeline.setError(`试听音色失败：${String(err)}`, book.bookId);
      pipeline.setStage("error", book.bookId);
    }
  }

  function chapterStatusIcon(chapter: ChapterMeta): string {
    if (pipeline.chapterAudioPaths[chapter.id]) return "✅";
    if (pipeline.analysis?.scriptPaths[chapter.id]) return "✓";
    return "—";
  }

  return (
    <main className="book-detail">
      <header className="detail-header">
        <button className="btn-back" onClick={onBack}>
          ← 返回书库
        </button>
        <h1>{book.title}</h1>
        <button
          className="btn-secondary"
          onClick={handleReextract}
          disabled={isBusy}
        >
          重新提取
        </button>
        <button className="btn-secondary" onClick={handleRegenerateAll}>
          全部重新生成
        </button>
        <button
          className="btn-primary"
          onClick={() => {
            if (pipeline.tab === "preview") pipeline.setTab("analyze", book.bookId);
            else if (pipeline.tab === "analyze") handleAnalyze();
            else if (pipeline.tab === "review") handleSaveCorrections();
            else handleGenerate();
          }}
          disabled={
            isBusy ||
            (pipeline.tab !== "preview" && pipeline.selectedChapters.size === 0)
          }
        >
          {pipeline.tab === "preview"
            ? "进入分析"
            : pipeline.tab === "analyze"
            ? `分析（${pipeline.selectedChapters.size}）`
            : pipeline.tab === "review"
              ? "保存修改"
              : `生成（${pipeline.selectedChapters.size}）`}
        </button>
      </header>

      <div className="detail-body">
        <aside className="chapter-list">
          <h3>章节</h3>
          <label className="select-all">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={toggleAllChapters}
            />
            <span>全选</span>
            <span className="select-all-count">
              已选 {pipeline.selectedChapters.size} / 共 {book.chapters.length} 章
            </span>
          </label>
          {book.chapters.map((ch) => (
            <label key={ch.id} className="chapter-item">
              <input
                type="checkbox"
                checked={pipeline.selectedChapters.has(ch.id)}
                onChange={() => toggleChapter(ch.id)}
              />
              <span className="chapter-status">
                {chapterStatusIcon(ch)}
              </span>
              <span className="chapter-title">{ch.title}</span>
            </label>
          ))}
        </aside>

        <section className="detail-content">
          <nav className="detail-tabs">
            <button
              className={`tab-btn ${pipeline.tab === "preview" ? "active" : ""}`}
              onClick={() => pipeline.setTab("preview", book.bookId)}
            >
              预览
            </button>
            <button
              className={`tab-btn ${pipeline.tab === "analyze" ? "active" : ""}`}
              onClick={() => pipeline.setTab("analyze", book.bookId)}
            >
              分析
            </button>
            <button
              className={`tab-btn ${pipeline.tab === "review" ? "active" : ""}`}
              onClick={() => pipeline.setTab("review", book.bookId)}
            >
              审阅
            </button>
            <button
              className={`tab-btn ${pipeline.tab === "generate" ? "active" : ""}`}
              onClick={() => pipeline.setTab("generate", book.bookId)}
            >
              生成
            </button>
          </nav>

          {/* Keep every audio element mounted while switching tabs. Hiding this
              container must not unmount the media elements, otherwise browser
              playback stops as soon as the user opens another tab. */}
          <div
            className={`audio-playback-persistence ${pipeline.tab === "generate" ? "is-visible" : "is-hidden"} ${listenableChapters.length > 0 ? "has-listenable-audio" : ""}`}
            aria-hidden={pipeline.tab !== "generate"}
          >
            {pipeline.tab === "generate" && listenableChapters.length > 0 && (
              <div className="listen-chapter-toolbar">
                <div>
                  <strong>试听章节</strong>
                  <span>只展示当前章节的原配音、混音、背景音乐和音效</span>
                </div>
                <label>
                  <span className="sr-only">选择试听章节</span>
                  <select
                    aria-label="选择试听章节"
                    value={activeListenChapterId ?? ""}
                    onChange={(event) => setListenChapterId(event.target.value || undefined)}
                    disabled={listenableChapters.length <= 1}
                  >
                    {listenableChapters.map((chapter) => (
                      <option key={chapter.id} value={chapter.id}>
                        {chapter.title}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
            <GeneratedAudioAssetList
              chapters={book.chapters}
              audioAssets={pipeline.audioAssets}
              activeChapterId={activeListenChapterId}
              onDownload={handleDownloadAudioAsset}
              onRegenerate={handleRegenerateAudioAsset}
              disabled={isBusy}
            />
            <PersistentChapterAudioList
              chapters={book.chapters}
              chapterAudioPaths={pipeline.chapterAudioPaths}
              chapterMixedAudioPaths={pipeline.chapterMixedAudioPaths}
              activeChapterId={activeListenChapterId}
              visible={pipeline.tab === "generate"}
              onDownload={handleDownloadChapter}
              onDownloadMixed={(chapterId, chapterTitle, path) =>
                handleDownloadChapter(chapterId, chapterTitle, path)
              }
              onRegenerate={handleRegenerateChapter}
              onRegenerateFinal={handleRegenerateChapter}
            />
          </div>

          {pipeline.tab === "preview" && (
            <ChapterPreview
              book={book}
              onContinue={() => pipeline.setTab("analyze", book.bookId)}
            />
          )}

          {pipeline.tab === "analyze" && (() => {
            const chapterMap = new Map(book.chapters.map((c) => [c.id, c]));
            const workflowEntries = Object.entries(pipeline.workflows.analysis).filter(([, workflow]) =>
              workflow.status !== "pending" || workflow.steps.some((step) => step.status !== "pending"),
            );
            const activityIds = new Set([
              ...Object.keys(pipeline.chapterStatuses),
              ...workflowEntries.map(([id]) => id),
            ]);
            const statusEntries = [...activityIds].map((id) => [
              id,
              pipeline.chapterStatuses[id] ?? pipeline.workflows.analysis[id]?.status ?? "pending",
            ] as const);
            const hasActivity = statusEntries.length > 0;
            const characters = pipeline.analysis?.characters ?? [];
            const activeAnalysisId = workflowEntries.find(([, workflow]) =>
              workflow.status === "running" || workflow.status === "needs_review" || workflow.status === "failed",
            )?.[0] ?? selectedChapterId ?? statusEntries[0]?.[0];
            const activeAnalysisWorkflow = activeAnalysisId
              ? pipeline.workflows.analysis[activeAnalysisId]
              : undefined;

            return (
              <div className="tab-panel analyze-panel">
                {/* Progress header — visible while busy */}
                {isBusy && (
                  <div className="analyze-progress-card">
                    <div className="analyze-progress-bar-wrap">
                      <div className="analyze-progress-track">
                        <div className="analyze-progress-fill" style={{ width: `${pipeline.progress}%` }} />
                      </div>
                      <span className="analyze-progress-pct">{pipeline.progress}%</span>
                    </div>
                    {pipeline.analyzeProgress && (
                      <div className="analyze-status-line">
                        <span className="spinner" />
                        <span>{pipeline.analyzeProgress}</span>
                      </div>
                    )}
                    {pipeline.progressDetail.length > 0 && (
                      <div className="analyze-detail-pills">
                        {pipeline.progressDetail.map((d) => (
                          <span key={d.label} className="analyze-detail-pill">
                            <span className="analyze-detail-label">{d.label}</span>
                            <span className="analyze-detail-value">{d.value}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    <button className="btn-secondary analyze-stop-btn" onClick={handleStop}>
                      停止
                    </button>
                  </div>
                )}

                {/* Idle ready state */}
                {!isBusy && !hasActivity && pipeline.selectedChapters.size > 0 && (
                  <div className="analyze-ready-state">
                    <span className="analyze-ready-icon">🔍</span>
                    <p>已选择 {pipeline.selectedChapters.size} 章</p>
                    <span className="analyze-ready-hint">点击“分析”提取角色并生成脚本。</span>
                  </div>
                )}

                {/* Post-run status: message only */}
                {!isBusy && pipeline.analyzeProgress && (
                  <p className="analyze-done-message">{pipeline.analyzeProgress}</p>
                )}

                {activeAnalysisWorkflow && (
                  <WorkflowSteps
                    workflow={activeAnalysisWorkflow}
                    title={`《${chapterMap.get(activeAnalysisWorkflow.chapterId)?.title ?? activeAnalysisWorkflow.chapterId}》分析阶段`}
                  />
                )}

                {/* Per-chapter status rows */}
                {hasActivity && (
                  <div className="analyze-chapter-list">
                    {statusEntries.map(([id, status]) => {
                      const ch = chapterMap.get(id);
                      const workflow = pipeline.workflows.analysis[id];
                      const isRunning = status === "analyzing" || workflow?.status === "running";
                      const isDone = status === "done" || workflow?.status === "succeeded";
                      const isFailed = status === "failed" || workflow?.status === "failed";
                      const currentStep = workflow?.steps.find((step) => step.id === workflow.currentStep);
                      const displayStatus = workflow
                        ? workflowStatusLabel(workflow.status)
                        : isRunning ? "分析中" : isDone ? "已完成" : isFailed ? "失败" : status;
                      return (
                        <div key={id} className={`analyze-chapter-row ${isRunning ? "is-running" : isDone ? "is-done" : isFailed ? "is-failed" : ""}`}>
                          <span className="chapter-status-dot cs-pending" style={
                            isRunning ? { background: "var(--accent)" } :
                            isDone    ? { background: "var(--success)" } :
                            isFailed  ? { background: "var(--danger)" } :
                            undefined
                          } />
                          <span className="analyze-chapter-name">{ch?.title ?? id}</span>
                          <span className={`status-chip ${isRunning ? "chip-running" : isDone ? "chip-done" : isFailed ? "chip-fail" : ""}`}>
                            {isRunning ? <><span className="spinner" /> {currentStep?.label ?? displayStatus}</> : displayStatus}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Characters found preview */}
                {characters.length > 0 && (
                  <div className="analyze-characters-preview">
                    <p className="analyze-section-label">已找到 {characters.length} 个角色</p>
                    <div className="analyze-char-chips">
                      {characters.map((c) => (
                        <span key={c.id} className="character-chip analyze-char-chip">
                          {c.canonicalName}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {pipeline.tab === "review" && pipeline.analysis && (
            <div className="tab-panel">
              <table className="character-table">
                <thead>
                  <tr>
                    <th>角色</th>
                    <th>MiMo 音色设计</th>
                    <th>性别</th>
                    <th>年龄</th>
                    <th>音色</th>
                    <th>试听</th>
                  </tr>
                </thead>
                <tbody>
                  <tr key="narrator">
                    <td style={{ fontWeight: 500 }}>旁白</td>
                    <td>固定旁白音色</td>
                    <td>
                      <select value={narratorVoiceId === "narrator_male" ? "male" : "female"} disabled aria-label="旁白性别">
                        <option value="female">女性</option>
                        <option value="male">男性</option>
                      </select>
                    </td>
                      <td>成年</td>
                    <td>
                      <select
                        value={narratorVoiceId === "narrator_male" ? "narrator_male" : "narrator_female"}
                        onChange={(event) => {
                          void handleNarratorVoiceChange(
                            event.target.value as "narrator_female" | "narrator_male",
                          );
                        }}
                        aria-label="旁白音色"
                      >
                        <option value="narrator_female">旁白（女性固定）</option>
                        <option value="narrator_male">旁白（男性固定）</option>
                      </select>
                    </td>
                    <td>
                      <button
                        onClick={() => handlePreviewVoice(narratorVoiceId)}
                        title="试听旁白音色"
                      >
                        ▶
                      </button>
                    </td>
                  </tr>
                  {pipeline.analysis.characters.map((c) => (
                    <tr key={c.id}>
                      <td style={{ fontWeight: 500 }}>
                        {c.canonicalName}
                        {c.identityStatus === "provisional" && (
                          <span className="status-chip chip-warning" style={{ marginLeft: 8 }}>
                            待确认
                          </span>
                        )}
                      </td>
                      <td>
                        <textarea
                          aria-label={`${c.canonicalName} 的 MiMo 音色设计`}
                          value={c.voiceDesign ?? ""}
                          rows={4}
                          onChange={(e) =>
                            handleVoiceDesignChange(c.id, e.target.value)
                          }
                          placeholder="角色身份、音色质感、说话习惯和固定约束"
                        />
                      </td>
                      <td>
                        <select
                          value={c.gender}
                          onChange={(e) =>
                            handleGenderChange(c.id, e.target.value)
                          }
                        >
                          <option value="male">男性</option>
                          <option value="female">女性</option>
                          <option value="neutral">中性</option>
                        </select>
                      </td>
                      <td>{ageLabel(c.ageClass)}</td>
                      <td>
                        <select
                          value={c.voiceId}
                          onChange={(e) =>
                            handleVoiceChange(c.id, e.target.value)
                          }
                        >
                          {voiceOptions
                            .filter(
                              (v) =>
                                !["narrator_default", "narrator_female", "narrator_male"].includes(v.id) ||
                                v.id === c.voiceId,
                            )
                            .map((v) => (
                            <option key={v.id} value={v.id} disabled={v.available === false}>
                              {localizeVoiceDisplayName(v.displayName, v.id)}{v.available === false ? "（不可用）" : ""}
                            </option>
                            ))}
                        </select>
                      </td>
                      <td>
                        <button
                          onClick={() => handlePreviewVoice(
                            c.voiceId,
                            c.voiceDesign ?? c.voiceDescription,
                            c.fallbackVoiceId,
                          )}
                          title="试听音色"
                        >
                          ▶
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <button
                  className="btn-primary"
                  style={{ width: "auto", padding: "8px 16px" }}
                  onClick={handleSaveCorrections}
                  disabled={!correctionState.dirty}
                >
                  保存修改
                </button>
                {pipeline.savedMessage && (
                  <span className="success-text" style={{ margin: 0 }}>
                    {pipeline.savedMessage}
                  </span>
                )}
              </div>
            </div>
          )}

          {pipeline.tab === "generate" && (
            <div className="tab-panel">
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                  选中的章节会在后端按顺序完成：原章节配音、Whisper 转录、音频规划、本地 Stable Audio 和最终混音。网页关闭或刷新后任务仍会继续。
                </div>
                {activeGenerationWorkflow && activeGenerationChapterId && (
                  <WorkflowSteps
                    workflow={activeGenerationWorkflow}
                    title={`《${activeGenerationTitle ?? activeGenerationChapterId}》音频流水线`}
                  />
                )}
                <button
                  className="btn-primary"
                  style={{ width: "auto", alignSelf: "flex-start" }}
                  onClick={handleGenerate}
                  disabled={
                    isBusy ||
                    !pipeline.analysis ||
                    pipeline.selectedChapters.size === 0
                  }
                >
                  批量生成（{pipeline.selectedChapters.size}）
                </button>
                {pipeline.selectedChapters.size === 0 && (
                  <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                    请在左侧选择一个或多个已经分析完成的章节。
                  </span>
                )}
              </div>
              {isBusy && (
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${pipeline.progress}%` }}
                  />
                </div>
              )}
              {pipeline.analyzeProgress && (
                <p style={{ margin: 0, fontSize: 12, color: "var(--text-secondary)" }}>
                  {pipeline.analyzeProgress}
                </p>
              )}
              {pipeline.progressDetail.map((d) => (
                <div key={d.label} className="progress-detail-row">
                  <span style={{ color: "var(--text-muted)" }}>{d.label}:</span> {d.value}
                </div>
              ))}
              {pipeline.savedMessage && (
                <p className="success-text" style={{ margin: 0 }}>
                  {pipeline.savedMessage}
                </p>
              )}
              <BatchGenerationStatusList batch={pipeline.generationBatch} />
              {!pipeline.generationBatch && Object.values(pipeline.workflows.generation).some((workflow) =>
                workflow.status !== "pending" || workflow.steps.some((step) => step.status !== "pending"),
              ) && (
                <div className="workflow-chapter-summary">
                  <p className="workflow-summary-title">已生成章节状态</p>
                  {book.chapters.filter((chapter) => {
                    const workflow = pipeline.workflows.generation[chapter.id];
                    return workflow && (
                      workflow.status !== "pending" || workflow.steps.some((step) => step.status !== "pending")
                    );
                  }).map((chapter) => {
                    const workflow = pipeline.workflows.generation[chapter.id];
                    if (!workflow) return null;
                    const currentStep = workflow.steps.find((step) => step.id === workflow.currentStep);
                    return (
                      <div className="workflow-summary-row" key={chapter.id}>
                        <span>{chapter.title}</span>
                        <span className={`workflow-summary-status workflow-${workflow.status}`}>
                          {currentStep?.label ?? workflowStatusLabel(workflow.status)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
              {isBusy && (
                <button className="btn-secondary" style={{ width: "auto", alignSelf: "flex-start" }} onClick={handleStop}>
                  停止
                </button>
              )}
            </div>
          )}

        </section>
      </div>

      {pipeline.error && (
        <div className="error-banner">
          <span>{pipeline.error}</span>
          <button
            onClick={() => {
              pipeline.setError(null, book.bookId);
              pipeline.setStage("idle", book.bookId);
            }}
          >
            关闭
          </button>
        </div>
      )}

      <footer
        className="character-strip"
        onClick={() => pipeline.setTab("review", book.bookId)}
        title="点击查看角色"
      >
        <span style={{ flexShrink: 0 }}>
          {pipeline.analysis
            ? `${pipeline.analysis.characters.length} 个角色`
            : "暂未发现角色"}
        </span>
        {pipeline.analysis &&
          pipeline.analysis.characters.slice(0, 5).map((c) => (
            <span key={c.id} className="character-chip">
              {c.canonicalName}
            </span>
          ))}
        {pipeline.analysis &&
          pipeline.analysis.characters.length > 5 && (
            <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
              另有 {pipeline.analysis.characters.length - 5} 个
            </span>
          )}
      </footer>
    </main>
  );
}
