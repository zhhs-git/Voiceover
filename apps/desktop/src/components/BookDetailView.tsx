import { useRef, useCallback, useEffect, useMemo, useState } from "react";
import { convertFileSrc, downloadFile, invoke } from "../lib/platform";
import type {
  AnalysisState,
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
import { buildVoiceOptions, localizeVoiceDisplayName } from "../lib/voiceOptions";
import { ChapterPreview } from "./ChapterPreview";
import { PersistentChapterAudioList } from "./PersistentChapterAudioList";
import {
  buildVoicePreviewRequest,
  buildVoicePreviewScript,
} from "../lib/voicePreview";

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
  const abortRef = useRef<AbortController | null>(null);
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
    setAnalysis: pipeline.setAnalysis,
    setCurrentStep: noopSetCurrentStep,
    setTab: pipeline.setTab,
    abortRef,
    db,
  });

  const { handleGenerate, handleRegenerateChapter, handleRegenerateAll } =
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
      setAnalyzeProgress: pipeline.setAnalyzeProgress,
      setProgressDetail: pipeline.setProgressDetail,
      setProgress: pipeline.setProgress,
      setChapterAudioPaths: pipeline.setChapterAudioPaths,
      setCurrentStep: noopSetCurrentStep,
      abortRef,
    });

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, [book.bookId]);

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
          } | null;
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

      // Fast bulk audio check — single invoke for all paths
      const audioPaths = book.chapters.map((ch) => `${book.workDir}/audio/${ch.id}.wav`);
      const existing: string[] = await invoke("file_exists", { paths: audioPaths });
      if (cancelled) return;
      const existingSet = new Set(existing);
      const paths: Record<string, string> = {};
      for (let i = 0; i < book.chapters.length; i++) {
        if (existingSet.has(audioPaths[i])) {
          paths[book.chapters[i].id] = audioPaths[i];
        }
      }
      pipeline.setChapterAudioPaths(paths, book.bookId);
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

  async function handleDownloadChapter(chapterId: string, chapterTitle: string) {
    const wavPath = pipeline.chapterAudioPaths[chapterId];
    if (!wavPath) return;
    const safeName = chapterTitle.replace(/[/\\:*?"<>|]/g, "_");
    const dest = `${book.workDir}/downloads/${safeName}.mp3`;
    pipeline.setSavedMessage(`正在转换"${chapterTitle}"…`, book.bookId);
    try {
      const result = await workerCall("convert_to_mp3", {
        wavPath,
        outputPath: dest,
      });
      if (result.status !== "succeeded") {
        throw new Error((result.error as any)?.message ?? "转换失败");
      }
      downloadFile(convertFileSrc(dest), `${safeName}.mp3`);
      pipeline.setSavedMessage(`"${chapterTitle}"已保存为 MP3。`, book.bookId);
    } catch (err) {
      pipeline.setError(`下载失败：${String(err)}`, book.bookId);
    }
  }

  function handleStop() {
    abortRef.current?.abort();
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

  async function handleSaveCorrections() {
    if (!book || !pipeline.analysis) return;
    pipeline.setStage("saving", book.bookId);
    pipeline.setError(null, book.bookId);
    pipeline.setSavedMessage(null, book.bookId);
    try {
      const result = await workerCall("apply_corrections", {
        bookId: book.bookId,
        chapters: book.chapters.map((c) => ({
          chapterId: c.id,
          textPath: c.textPath,
          title: c.title,
        })),
        corrections: {
          aliasMerges: correctionState.aliasMerges,
          genderOverrides: correctionState.genderOverrides,
          voiceOverrides: correctionState.voiceOverrides,
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
        buildVoicePreviewRequest(scriptPath, previewDir, segmentId),
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
      const audio = new Audio(convertFileSrc(audioPath));
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
            全选
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

          {pipeline.tab === "preview" && (
            <ChapterPreview
              book={book}
              onContinue={() => pipeline.setTab("analyze", book.bookId)}
            />
          )}

          {pipeline.tab === "analyze" && (() => {
            const chapterMap = new Map(book.chapters.map((c) => [c.id, c]));
            const statusEntries = Object.entries(pipeline.chapterStatuses);
            const hasActivity = statusEntries.length > 0;
            const characters = pipeline.analysis?.characters ?? [];

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

                {/* Per-chapter status rows */}
                {hasActivity && (
                  <div className="analyze-chapter-list">
                    {statusEntries.map(([id, status]) => {
                      const ch = chapterMap.get(id);
                      const isRunning = status === "analyzing";
                      const isDone = status === "done";
                      const isFailed = status === "failed";
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
                            {isRunning ? <><span className="spinner" /> 分析中</> : isDone ? "已完成" : isFailed ? "失败" : status}
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
                    <th>性别</th>
                    <th>年龄</th>
                    <th>音色</th>
                    <th>试听</th>
                  </tr>
                </thead>
                <tbody>
                  <tr key="narrator">
                    <td style={{ fontWeight: 500 }}>旁白</td>
                    <td>
                      <select value="neutral" disabled aria-label="旁白性别">
                        <option value="neutral">中性</option>
                      </select>
                    </td>
                      <td>成年</td>
                    <td>
                      <select value="narrator_default" disabled aria-label="旁白音色">
                        <option value="narrator_default">旁白（固定）</option>
                      </select>
                    </td>
                    <td>
                      <button
                        onClick={() => handlePreviewVoice("narrator_default")}
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
                          {voiceOptions.map((v) => (
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
                            c.voiceDescription,
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
              {isBusy && (
                <button className="btn-secondary" style={{ width: "auto", alignSelf: "flex-start" }} onClick={handleStop}>
                  停止
                </button>
              )}
            </div>
          )}

          <PersistentChapterAudioList
            chapters={book.chapters}
            chapterAudioPaths={pipeline.chapterAudioPaths}
            visible={pipeline.tab === "generate"}
            onDownload={handleDownloadChapter}
            onRegenerate={handleRegenerateChapter}
          />
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
