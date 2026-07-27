import { useCallback } from "react";
import { invoke } from "../lib/platform";

import type {
  AnalysisState,
  BookState,
  CharacterMeta,
  PipelineStage,
  ProgressDetail,
  WorkspaceStep,
} from "../types";
import { workerCall } from "../lib/workerCall";
import type { DetailTab } from "../state/pipelineStore";

interface UseChapterAnalysisDeps {
  book: BookState | null;
  analysis: AnalysisState | null;
  selectedChapters: Set<string>;
  setStage: (stage: PipelineStage, owner?: string) => void;
  setError: (error: string | null, owner?: string) => void;
  setSavedMessage: (msg: string | null, owner?: string) => void;
  setAnalyzeProgress: (msg: string, owner?: string) => void;
  setChapterStatuses: (statuses: Record<string, string>, owner?: string) => void;
  setProgressDetail: (
    details: ProgressDetail[] | ((prev: ProgressDetail[]) => ProgressDetail[]),
    owner?: string,
  ) => void;
  setProgress: (progress: number, owner?: string) => void;
  setAnalysis: (
    analysis:
      | AnalysisState
      | null
      | ((prev: AnalysisState | null) => AnalysisState | null),
    owner?: string,
  ) => void;
  setCurrentStep: (step: WorkspaceStep) => void;
  setTab: (tab: DetailTab, owner?: string) => void;
  abortRef: React.MutableRefObject<AbortController | null>;
  db: {
    upsertChapter: (record: {
      id: string;
      bookId: string;
      title: string;
      status: string;
      scriptPath?: string;
    }) => Promise<unknown>;
    upsertCharacter: (record: {
      id: string; bookId: string; canonicalName: string;
      gender?: string | null; ageClass?: string | null;
      identityStatus?: "provisional" | "confirmed" | "merged" | null;
      voiceId?: string | null; voiceSource?: "auto" | "manual" | null;
      voiceAssignmentVersion?: number | null; voiceProfile?: string | null;
      fallbackVoiceId?: string | null;
      voiceDescription?: string | null;
      confidence?: number; aliases?: string;
    }) => Promise<unknown>;
  };
}

export function useChapterAnalysis(deps: UseChapterAnalysisDeps) {
  const {
    book,
    analysis,
    selectedChapters,
    setStage,
    setError,
    setSavedMessage,
    setAnalyzeProgress,
    setChapterStatuses,
    setProgressDetail,
    setProgress,
    setAnalysis,
    setCurrentStep,
    setTab,
    abortRef,
    db,
  } = deps;

  const handleAnalyze = useCallback(async () => {
    if (!book) return;
    const controller = new AbortController();
    abortRef.current = controller;
    const clearController = () => {
      if (abortRef.current === controller) abortRef.current = null;
    };
    setStage("analyzing", book.bookId);
    setError(null, book.bookId);
    setSavedMessage(null, book.bookId);
    setProgress(0, book.bookId);
    setAnalyzeProgress("正在开始分析…", book.bookId);
    setChapterStatuses({}, book.bookId);
    setProgressDetail([{ label: "模型", value: "DeepSeek Flash" }], book.bookId);

    const startTime = Date.now();
    const modelLabel = "DeepSeek Flash";

    const chaptersToAnalyze = book.chapters.filter((c) =>
      selectedChapters.has(c.id),
    );

    // Characters accumulated across this run (seeded with previously-found ones).
    // Each chapter receives the full list so the LLM can maintain consistency.
    type KnownChar = {
      id: string;
      canonicalName: string;
      aliases: string[];
      gender: string;
      ageClass?: string;
      identityStatus?: "provisional" | "confirmed" | "merged";
      voiceId?: string | null;
      voiceSource?: "auto" | "manual" | null;
      voiceAssignmentVersion?: number | null;
      voiceProfile?: string | null;
      fallbackVoiceId?: string | null;
      voiceDescription?: string | null;
      confidence?: number;
    };
    // The complete book-scoped roster is the source of truth. Voice fields are
    // passed through so a single-chapter re-analysis cannot erase a manual
    // assignment while the worker resolves character identity.
    const knownCharacters: KnownChar[] = (analysis?.characters ?? []).map((c) => ({
      id: c.id,
      canonicalName: c.canonicalName,
      aliases: c.aliases ?? [],
      gender: c.gender,
      ageClass: c.ageClass,
      identityStatus: c.identityStatus,
      voiceId: c.voiceId,
      voiceSource: c.voiceSource,
      voiceAssignmentVersion: c.voiceAssignmentVersion,
      voiceProfile: c.voiceProfile,
      fallbackVoiceId: c.fallbackVoiceId,
      voiceDescription: c.voiceDescription,
      confidence: c.confidence,
    }));

    // Track elapsed time alongside chapter progress
    const elapsedTimer = setInterval(() => {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      setProgressDetail((prev: ProgressDetail[]) => {
        const next = prev.filter((d: ProgressDetail) => d.label !== "已用时");
        return [...next, { label: "已用时", value: `${elapsed} 秒` }];
      }, book.bookId);
    }, 1000);

    try {
      const statuses: Record<string, string> = {};
      let doneCount = 0;

      for (let i = 0; i < chaptersToAnalyze.length; i++) {
        if (controller.signal.aborted) break;
        const chapter = chaptersToAnalyze[i];

        // Progress: 0 → 100% linearly as chapters complete
        setProgress(Math.round((i / chaptersToAnalyze.length) * 100), book.bookId);
        setAnalyzeProgress(
          `正在分析第 ${i + 1} 章（共 ${chaptersToAnalyze.length} 章）…`,
          book.bookId,
        );
        setProgressDetail([
          { label: "模型", value: modelLabel },
          { label: "进度", value: `${i + 1} / ${chaptersToAnalyze.length}` },
          {
            label: "当前章节",
            value: chapter.title.length > 30
              ? chapter.title.slice(0, 30) + "…"
              : chapter.title,
          },
          { label: "已用时", value: `${Math.round((Date.now() - startTime) / 1000)} 秒` },
        ], book.bookId);
        statuses[chapter.id] = "analyzing";
        setChapterStatuses({ ...statuses }, book.bookId);

        try {
          const result = await workerCall("analyze_chapter", {
            bookId: book.bookId,
            chapterId: chapter.id,
            title: chapter.title,
            chapterTextPath: chapter.textPath,
            outputDirectory: `${book.workDir}/scripts`,
            // Pass accumulated character context for cross-chapter consistency
            knownCharacters: knownCharacters.length > 0 ? knownCharacters : undefined,
          });

          if (result.status !== "succeeded") {
            statuses[chapter.id] = "failed";
            setChapterStatuses({ ...statuses }, book.bookId);
            continue;
          }

          const artifact = (result.artifacts as Array<{ path: string }>)[0];
          statuses[chapter.id] = "done";
          doneCount++;

          db.upsertChapter({
            id: chapter.id,
            bookId: book.bookId,
            title: chapter.title,
            status: "succeeded",
            scriptPath: artifact.path,
          }).catch(() => {});

          // Read script to extract characters
          const scriptRaw = await invoke<string>("run_worker", {
            command: "_read_file",
            inputJson: JSON.stringify({ path: artifact.path }),
          }).catch(() => "{}");

          const scriptData = JSON.parse(scriptRaw) as {
            characters?: CharacterMeta[];
            voices?: AnalysisState["voices"];
          } | null;

          const newChars = scriptData?.characters ?? [];
          const newVoices = scriptData?.voices ?? [];

          // Add newly discovered characters to the running context for next chapters
          for (const c of newChars) {
            const existing = knownCharacters.find((item) => item.id === c.id);
            if (!existing) {
              knownCharacters.push({
                id: c.id,
                canonicalName: c.canonicalName,
                aliases: c.aliases ?? [],
                gender: c.gender,
                ageClass: c.ageClass,
                identityStatus: c.identityStatus,
                voiceId: c.voiceId,
                voiceSource: c.voiceSource,
                voiceAssignmentVersion: c.voiceAssignmentVersion,
                voiceProfile: c.voiceProfile,
                fallbackVoiceId: c.fallbackVoiceId,
                voiceDescription: c.voiceDescription,
                confidence: c.confidence,
              });
              continue;
            }
            existing.aliases = [...new Set([...(existing.aliases ?? []), ...(c.aliases ?? [])])];
            if (!existing.gender || existing.gender === "unknown") existing.gender = c.gender;
            if (!existing.ageClass || existing.ageClass === "unknown") existing.ageClass = c.ageClass;
            if (c.identityStatus === "confirmed" || !existing.identityStatus) {
              existing.identityStatus = c.identityStatus;
            }
            if (existing.voiceSource !== "manual") {
              existing.voiceId = c.voiceId;
              existing.voiceSource = c.voiceSource;
              existing.voiceAssignmentVersion = c.voiceAssignmentVersion;
              existing.voiceProfile = c.voiceProfile;
              existing.fallbackVoiceId = c.fallbackVoiceId;
              existing.voiceDescription = c.voiceDescription;
            }
          }

          // Persist characters to DB before exposing the result to later chapters.
          // Awaiting this keeps the durable roster in sync with knownCharacters.
          for (const c of newChars) {
            await db.upsertCharacter({
              id: c.id,
              bookId: book.bookId,
              canonicalName: c.canonicalName,
              gender: c.gender,
              ageClass: c.ageClass,
              identityStatus: c.identityStatus,
              voiceId: c.voiceId,
              voiceSource: c.voiceSource,
              voiceAssignmentVersion: c.voiceAssignmentVersion,
              voiceProfile: c.voiceProfile,
              fallbackVoiceId: c.fallbackVoiceId,
              voiceDescription: c.voiceDescription,
              confidence: c.confidence,
              aliases: JSON.stringify(c.aliases),
            });
          }

          // Merge into existing analysis state in real-time
          setAnalysis((prev) => {
            const existingCharIds = new Set(
              (prev?.characters ?? []).map((c) => c.id),
            );
            const existingVoiceIds = new Set(
              (prev?.voices ?? []).map((v) => v.id),
            );
            const incomingById = new Map(newChars.map((character) => [character.id, character]));
            const mergedCharacters = (prev?.characters ?? []).map((character) => {
              const incoming = incomingById.get(character.id);
              if (!incoming) return character;
              if (character.voiceSource === "manual") {
                return {
                  ...character,
                  ...incoming,
                  voiceId: character.voiceId,
                  voiceSource: character.voiceSource,
                  voiceAssignmentVersion: character.voiceAssignmentVersion,
                  voiceProfile: character.voiceProfile,
                  fallbackVoiceId: character.fallbackVoiceId,
                  voiceDescription: character.voiceDescription,
                };
              }
              return { ...character, ...incoming };
            });
            return {
              characters: [
                ...mergedCharacters,
                ...newChars.filter((c) => !existingCharIds.has(c.id)),
              ],
              voices: [
                ...(prev?.voices ?? []),
                ...newVoices.filter((v) => !existingVoiceIds.has(v.id)),
              ],
              scriptPaths: {
                ...(prev?.scriptPaths ?? {}),
                [chapter.id]: artifact.path,
              },
            };
          }, book.bookId);
        } catch {
          statuses[chapter.id] = "failed";
        }

        setChapterStatuses({ ...statuses }, book.bookId);
      }

      clearInterval(elapsedTimer);
      const wasStopped = controller.signal.aborted;
      const failedCount = Object.values(statuses).filter((s) => s === "failed").length;

      setProgress(100, book.bookId);
      setProgressDetail([], book.bookId);
        setAnalyzeProgress(
          wasStopped
          ? `已停止，完成 ${doneCount} / ${chaptersToAnalyze.length} 章。`
          : failedCount > 0
            ? `分析完成：${doneCount} 章成功，${failedCount} 章失败。`
            : `已分析 ${doneCount} 章。`,
        book.bookId,
      );
      setStage("idle", book.bookId);
      clearController();

      if (doneCount > 0) {
        setCurrentStep(3);
        // Auto-advance to review tab so user can see characters
        setTab("review", book.bookId);
      }
    } catch (err) {
      clearInterval(elapsedTimer);
      if (!controller.signal.aborted) {
        setError(`分析失败：${String(err)}`, book.bookId);
        setStage("error", book.bookId);
        clearController();
      } else {
        setProgress(100, book.bookId);
        setProgressDetail([], book.bookId);
        setAnalyzeProgress("分析已停止。", book.bookId);
        setStage("idle", book.bookId);
        clearController();
      }
    }
  }, [
    book,
    analysis,
    selectedChapters,
    abortRef,
    db,
    setAnalysis,
    setAnalyzeProgress,
    setChapterStatuses,
    setCurrentStep,
    setError,
    setProgress,
    setProgressDetail,
    setSavedMessage,
    setStage,
    setTab,
  ]);

  return { handleAnalyze };
}
