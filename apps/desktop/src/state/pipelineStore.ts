import { create } from "zustand";
import type {
  AnalysisState,
  PipelineStage,
  ProgressDetail,
} from "../types";

export type DetailTab = "preview" | "analyze" | "review" | "generate";

type BookOwner = string | undefined;

type AnalysisUpdate =
  | AnalysisState
  | null
  | ((prev: AnalysisState | null) => AnalysisState | null);

type PathsUpdate =
  | Record<string, string>
  | ((prev: Record<string, string>) => Record<string, string>);

type DetailsUpdate =
  | ProgressDetail[]
  | ((prev: ProgressDetail[]) => ProgressDetail[]);

type ChaptersUpdate = Set<string> | ((prev: Set<string>) => Set<string>);

function canUpdate(bookId: string | null, owner: BookOwner): boolean {
  return owner === undefined || owner === bookId;
}

function resetValues(bookId: string | null) {
  return {
    bookId,
    analysis: null,
    chapterAudioPaths: {},
    stage: "idle" as PipelineStage,
    error: null,
    progress: 0,
    savedMessage: null,
    analyzeProgress: "",
    chapterStatuses: {},
    progressDetail: [],
    selectedChapters: new Set<string>(),
    tab: "preview" as DetailTab,
  };
}

interface PipelineState {
  bookId: string | null;
  analysis: AnalysisState | null;
  chapterAudioPaths: Record<string, string>;
  stage: PipelineStage;
  error: string | null;
  progress: number;
  savedMessage: string | null;
  analyzeProgress: string;
  chapterStatuses: Record<string, string>;
  progressDetail: ProgressDetail[];
  selectedChapters: Set<string>;
  tab: DetailTab;

  activateBook: (bookId: string) => void;
  setAnalysis: (analysis: AnalysisUpdate, owner?: BookOwner) => void;
  setChapterAudioPaths: (paths: PathsUpdate, owner?: BookOwner) => void;
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
