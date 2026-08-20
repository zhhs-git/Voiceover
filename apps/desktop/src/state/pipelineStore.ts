import { create } from "zustand";
import type {
  AnalysisState,
  AudioAsset,
  ChapterWorkflowStatus,
  PipelineStage,
  ProgressDetail,
} from "../types";
import type { BatchGenerationResponse } from "../lib/batchGeneration";
import type { WorkflowKind } from "../lib/workflowStatus";

export type DetailTab =
  | "preview"
  | "analyze"
  | "review"
  | "generate"
  | "download"
  | "model-settings";

type BookOwner = string | undefined;

type AnalysisUpdate =
  | AnalysisState
  | null
  | ((prev: AnalysisState | null) => AnalysisState | null);

type PathsUpdate =
  | Record<string, string>
  | ((prev: Record<string, string>) => Record<string, string>);

type AudioAssetsUpdate =
  | Record<string, AudioAsset[]>
  | ((prev: Record<string, AudioAsset[]>) => Record<string, AudioAsset[]>);

type BatchGenerationUpdate =
  | BatchGenerationResponse
  | null
  | ((prev: BatchGenerationResponse | null) => BatchGenerationResponse | null);

type DetailsUpdate =
  | ProgressDetail[]
  | ((prev: ProgressDetail[]) => ProgressDetail[]);

type ChaptersUpdate = Set<string> | ((prev: Set<string>) => Set<string>);

type WorkflowStatusesUpdate =
  | Record<string, ChapterWorkflowStatus>
  | ((prev: Record<string, ChapterWorkflowStatus>) => Record<string, ChapterWorkflowStatus>);

function canUpdate(bookId: string | null, owner: BookOwner): boolean {
  return owner === undefined || owner === bookId;
}

function keepNewerWorkflow(
  current: ChapterWorkflowStatus | undefined,
  incoming: ChapterWorkflowStatus,
): ChapterWorkflowStatus {
  if (
    current &&
    typeof current.updatedAt === "number" &&
    (typeof incoming.updatedAt !== "number" || current.updatedAt >= incoming.updatedAt)
  ) {
    return current;
  }
  return incoming;
}

function resetValues(bookId: string | null) {
  return {
    bookId,
    analysis: null,
    chapterAudioPaths: {},
    chapterMixedAudioPaths: {},
    audioAssets: {},
    generationBatch: null,
    stage: "idle" as PipelineStage,
    error: null,
    progress: 0,
    savedMessage: null,
    analyzeProgress: "",
    chapterStatuses: {},
    progressDetail: [],
    workflows: { analysis: {}, generation: {} },
    selectedChapters: new Set<string>(),
    tab: "preview" as DetailTab,
  };
}

interface PipelineState {
  bookId: string | null;
  analysis: AnalysisState | null;
  chapterAudioPaths: Record<string, string>;
  chapterMixedAudioPaths: Record<string, string>;
  audioAssets: Record<string, AudioAsset[]>;
  /** Durable backend queue status for the most recent generation batch. */
  generationBatch: BatchGenerationResponse | null;
  stage: PipelineStage;
  error: string | null;
  progress: number;
  savedMessage: string | null;
  analyzeProgress: string;
  chapterStatuses: Record<string, string>;
  progressDetail: ProgressDetail[];
  workflows: {
    analysis: Record<string, ChapterWorkflowStatus>;
    generation: Record<string, ChapterWorkflowStatus>;
  };
  selectedChapters: Set<string>;
  tab: DetailTab;

  activateBook: (bookId: string) => void;
  setAnalysis: (analysis: AnalysisUpdate, owner?: BookOwner) => void;
  setChapterAudioPaths: (paths: PathsUpdate, owner?: BookOwner) => void;
  setChapterMixedAudioPaths: (paths: PathsUpdate, owner?: BookOwner) => void;
  setAudioAssets: (assets: AudioAssetsUpdate, owner?: BookOwner) => void;
  setGenerationBatch: (batch: BatchGenerationUpdate, owner?: BookOwner) => void;
  setStage: (stage: PipelineStage, owner?: BookOwner) => void;
  setError: (error: string | null, owner?: BookOwner) => void;
  setProgress: (progress: number, owner?: BookOwner) => void;
  setSavedMessage: (msg: string | null, owner?: BookOwner) => void;
  setAnalyzeProgress: (msg: string, owner?: BookOwner) => void;
  setChapterStatuses: (
    statuses: Record<string, string>,
    owner?: BookOwner,
  ) => void;
  setProgressDetail: (details: DetailsUpdate, owner?: BookOwner) => void;
  setWorkflowStatus: (
    kind: WorkflowKind,
    chapterId: string,
    status: ChapterWorkflowStatus,
    owner?: BookOwner,
  ) => void;
  setWorkflowStatuses: (
    kind: WorkflowKind,
    statuses: WorkflowStatusesUpdate,
    owner?: BookOwner,
  ) => void;
  setSelectedChapters: (chapters: ChaptersUpdate, owner?: BookOwner) => void;
  setTab: (tab: DetailTab, owner?: BookOwner) => void;
  resetPipeline: (bookId?: string | null) => void;
}

export const usePipelineStore = create<PipelineState>((set) => ({
  ...resetValues(null),

  activateBook: (bookId) =>
    set((state) =>
      state.bookId === bookId ? {} : resetValues(bookId),
    ),

  setAnalysis: (analysis, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        analysis:
          typeof analysis === "function" ? analysis(state.analysis) : analysis,
      };
    }),

  setChapterAudioPaths: (paths, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        chapterAudioPaths:
          typeof paths === "function"
            ? paths(state.chapterAudioPaths)
            : paths,
      };
    }),
  setChapterMixedAudioPaths: (paths, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        chapterMixedAudioPaths:
          typeof paths === "function"
            ? paths(state.chapterMixedAudioPaths)
            : paths,
      };
    }),
  setAudioAssets: (assets, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        audioAssets:
          typeof assets === "function" ? assets(state.audioAssets) : assets,
      };
    }),
  setGenerationBatch: (batch, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        generationBatch:
          typeof batch === "function" ? batch(state.generationBatch) : batch,
      };
    }),

  setStage: (stage, owner) =>
    set((state) => (canUpdate(state.bookId, owner) ? { stage } : {})),
  setError: (error, owner) =>
    set((state) => (canUpdate(state.bookId, owner) ? { error } : {})),
  setProgress: (progress, owner) =>
    set((state) => (canUpdate(state.bookId, owner) ? { progress } : {})),
  setSavedMessage: (savedMessage, owner) =>
    set((state) =>
      canUpdate(state.bookId, owner) ? { savedMessage } : {},
    ),
  setAnalyzeProgress: (analyzeProgress, owner) =>
    set((state) =>
      canUpdate(state.bookId, owner) ? { analyzeProgress } : {},
    ),
  setChapterStatuses: (chapterStatuses, owner) =>
    set((state) =>
      canUpdate(state.bookId, owner) ? { chapterStatuses } : {},
    ),
  setProgressDetail: (details, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        progressDetail:
          typeof details === "function"
            ? details(state.progressDetail)
            : details,
      };
    }),
  setWorkflowStatus: (kind, chapterId, status, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        workflows: {
          ...state.workflows,
          [kind]: {
            ...state.workflows[kind],
            [chapterId]: keepNewerWorkflow(state.workflows[kind][chapterId], status),
          },
        },
      };
    }),
  setWorkflowStatuses: (kind, statuses, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        workflows: {
          ...state.workflows,
          [kind]: typeof statuses === "function"
            ? statuses(state.workflows[kind])
            : statuses,
        },
      };
    }),
  setSelectedChapters: (chapters, owner) =>
    set((state) => {
      if (!canUpdate(state.bookId, owner)) return {};
      return {
        selectedChapters:
          typeof chapters === "function"
            ? chapters(state.selectedChapters)
            : chapters,
      };
    }),
  setTab: (tab, owner) =>
    set((state) => (canUpdate(state.bookId, owner) ? { tab } : {})),
  resetPipeline: (bookId = null) => set(resetValues(bookId)),
}));
