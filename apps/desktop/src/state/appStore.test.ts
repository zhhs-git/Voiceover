// @vitest-environment node
import { beforeEach, describe, expect, test } from "vitest";
import type { BookState } from "../types";
import { useAppStore } from "./appStore";
import { useCorrectionStore } from "./corrections";
import { usePipelineStore } from "./pipelineStore";

const makeBook = (bookId: string): BookState => ({
  title: bookId,
  bookId,
  workDir: `/books/${bookId}`,
  chapters: [],
});

describe("app book navigation", () => {
  beforeEach(() => {
    usePipelineStore.getState().resetPipeline();
    useCorrectionStore.getState().reset();
  });

  test("switching books resets analysis and pending corrections", () => {
    useAppStore.getState().navigateToBook(makeBook("book-pei-yin"), "/pei-yin.txt");
    usePipelineStore.getState().setAnalysis({
      characters: [],
      voices: [],
      scriptPaths: {},
    }, "book-pei-yin");
    useCorrectionStore.getState().setGender("character_a", "female");

    useAppStore.getState().navigateToBook(makeBook("book-han-guang"), "/han-guang.txt");

    expect(usePipelineStore.getState().bookId).toBe("book-han-guang");
    expect(usePipelineStore.getState().analysis).toBeNull();
    expect(useCorrectionStore.getState().genderOverrides).toEqual([]);
  });

  test("returning to the library clears pending corrections", () => {
    useAppStore.getState().navigateToBook(makeBook("book-pei-yin"), "/pei-yin.txt");
    useCorrectionStore.getState().setVoice("character_a", "male_adult_01");

    useAppStore.getState().navigateToLibrary();

    expect(useCorrectionStore.getState().voiceOverrides).toEqual([]);
  });
});
