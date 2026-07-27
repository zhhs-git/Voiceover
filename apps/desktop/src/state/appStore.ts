import { create } from "zustand";
import type { AppView, BookState } from "../types";
import { useCorrectionStore } from "./corrections";
import { usePipelineStore } from "./pipelineStore";

interface AppState {
  view: AppView;
  activeBook: BookState | null;
  activeSourcePath: string;
  importError: string | null;
  navigateToLibrary: () => void;
  navigateToBook: (book: BookState, sourcePath: string) => void;
  setImportError: (err: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  view: { page: "library" },
  activeBook: null,
  activeSourcePath: "",
  importError: null,
  navigateToLibrary: () => {
    usePipelineStore.getState().resetPipeline();
    useCorrectionStore.getState().reset();
    set({ view: { page: "library" }, importError: null });
  },
  navigateToBook: (book, sourcePath) => {
    // The pipeline is global, so make the active book boundary explicit
    // before React renders the next detail view.
    const pipeline = usePipelineStore.getState();
    if (pipeline.bookId !== book.bookId) {
      pipeline.activateBook(book.bookId);
      useCorrectionStore.getState().reset();
    }
    set({
      activeBook: book,
      activeSourcePath: sourcePath,
      view: { page: "bookDetail", bookId: book.bookId },
    });
  },
  setImportError: (err) => set({ importError: err }),
}));
