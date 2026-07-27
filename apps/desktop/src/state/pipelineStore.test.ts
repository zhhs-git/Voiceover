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
});
