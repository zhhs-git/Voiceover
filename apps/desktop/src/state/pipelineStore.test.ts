// @vitest-environment node
import { beforeEach, describe, expect, test } from "vitest";
import { usePipelineStore } from "./pipelineStore";

const character = {
  id: "character_lingxiu",
  canonicalName: "灵秀",
  aliases: [],
  gender: "female",
  voiceId: "female_adult_01",
  confidence: 0.95,
};

describe("book-scoped pipeline state", () => {
  beforeEach(() => {
    usePipelineStore.getState().resetPipeline();
  });

  test("switching books clears the previous book snapshot", () => {
    const store = usePipelineStore.getState();
    store.activateBook("book_pei-yin");
    store.setAnalysis({
      characters: [character],
      voices: [],
      scriptPaths: { chapter_001: "/books/pei-yin/chapter_001.json" },
    }, "book_pei-yin");
    store.setChapterAudioPaths(
      { chapter_001: "/books/pei-yin/chapter_001.wav" },
      "book_pei-yin",
    );
    store.setSelectedChapters(new Set(["chapter_001"]), "book_pei-yin");
    store.setTab("review", "book_pei-yin");

    usePipelineStore.getState().activateBook("book-han-guang");
    const next = usePipelineStore.getState();

    expect(next.bookId).toBe("book-han-guang");
    expect(next.analysis).toBeNull();
    expect(next.chapterAudioPaths).toEqual({});
    expect(next.selectedChapters).toEqual(new Set());
    expect(next.tab).toBe("preview");
  });

  test("a late update from the previous book is ignored", () => {
    const store = usePipelineStore.getState();
    store.activateBook("book_pei-yin");
    usePipelineStore.getState().activateBook("book-han-guang");

    store.setAnalysis(
      {
        characters: [character],
        voices: [],
        scriptPaths: { chapter_001: "/books/pei-yin/chapter_001.json" },
      },
      "book_pei-yin",
    );
    store.setChapterAudioPaths(
      { chapter_001: "/books/pei-yin/chapter_001.wav" },
      "book_pei-yin",
    );
    store.setAnalyzeProgress("old book finished", "book_pei-yin");

    const current = usePipelineStore.getState();
    expect(current.analysis).toBeNull();
    expect(current.chapterAudioPaths).toEqual({});
    expect(current.analyzeProgress).toBe("");
  });

  test("a late restore can fill missing data without replacing newer chapter results", () => {
    const store = usePipelineStore.getState();
    store.activateBook("book_pei-yin");
    store.setChapterAudioPaths(
      { chapter_new: "/books/pei-yin/audio/chapter_new.wav" },
      "book_pei-yin",
    );
    store.setWorkflowStatuses("generation", {
      chapter_new: {
        chapterId: "chapter_new",
        kind: "generation",
        status: "succeeded",
        steps: [],
      },
    }, "book_pei-yin");

    store.setChapterAudioPaths(
      (current) => ({
        chapter_old: "/books/pei-yin/audio/chapter_old.wav",
        ...current,
      }),
      "book_pei-yin",
    );
    store.setWorkflowStatuses("generation", (current) => ({
      chapter_old: {
        chapterId: "chapter_old",
        kind: "generation",
        status: "pending",
        steps: [],
      },
      ...current,
    }), "book_pei-yin");

    const current = usePipelineStore.getState();
    expect(current.chapterAudioPaths).toEqual({
      chapter_old: "/books/pei-yin/audio/chapter_old.wav",
      chapter_new: "/books/pei-yin/audio/chapter_new.wav",
    });
    expect(current.workflows.generation.chapter_new.status).toBe("succeeded");
    expect(current.workflows.generation.chapter_old.status).toBe("pending");
  });

  test("an older workflow poll cannot overwrite a newer terminal batch result", () => {
    const store = usePipelineStore.getState();
    store.activateBook("book_pei-yin");
    store.setWorkflowStatus("generation", "chapter_001", {
      chapterId: "chapter_001",
      kind: "generation",
      status: "succeeded",
      steps: [],
      updatedAt: 200,
    }, "book_pei-yin");

    store.setWorkflowStatus("generation", "chapter_001", {
      chapterId: "chapter_001",
      kind: "generation",
      status: "running",
      steps: [],
      updatedAt: 100,
    }, "book_pei-yin");

    expect(usePipelineStore.getState().workflows.generation.chapter_001).toMatchObject({
      status: "succeeded",
      updatedAt: 200,
    });
  });
});
