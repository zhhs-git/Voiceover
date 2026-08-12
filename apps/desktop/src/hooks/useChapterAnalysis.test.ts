import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { invoke } from "../lib/platform";
import { workerCall } from "../lib/workerCall";
import { watchChapterWorkflow } from "../lib/workflowStatus";
import { useChapterAnalysis } from "./useChapterAnalysis";

vi.mock("../lib/platform", () => ({
  invoke: vi.fn(),
}));

vi.mock("../lib/workerCall", () => ({
  workerCall: vi.fn(),
}));

vi.mock("../lib/workflowStatus", () => ({
  watchChapterWorkflow: vi.fn(),
}));

describe("useChapterAnalysis", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(watchChapterWorkflow).mockReturnValue(vi.fn());
    vi.mocked(invoke).mockResolvedValue(
      JSON.stringify({ characters: [], voices: [] }),
    );
    vi.mocked(workerCall).mockResolvedValue({
      status: "succeeded",
      artifacts: [{ kind: "chapter_script", path: "/tmp/chapter.json" }],
    } as never);
  });

  it("does not synthesize audio while re-analyzing a chapter with an existing script", async () => {
    const deps = {
      book: {
        title: "Book",
        bookId: "book_123",
        workDir: "/tmp/book",
        chapters: [
          {
            id: "chapter_001",
            title: "第一章",
            textLength: 100,
            textPath: "/tmp/chapter.txt",
          },
        ],
      },
      analysis: {
        characters: [],
        voices: [],
        scriptPaths: { chapter_001: "/tmp/old-chapter.json" },
      },
      selectedChapters: new Set(["chapter_001"]),
      setStage: vi.fn(),
      setError: vi.fn(),
      setSavedMessage: vi.fn(),
      setAnalyzeProgress: vi.fn(),
      setChapterStatuses: vi.fn(),
      setProgressDetail: vi.fn(),
      setProgress: vi.fn(),
      setWorkflowStatus: vi.fn(),
      setAnalysis: vi.fn(),
      setChapterAudioPaths: vi.fn(),
      setChapterMixedAudioPaths: vi.fn(),
      setAudioAssets: vi.fn(),
      setCurrentStep: vi.fn(),
      setTab: vi.fn(),
      abortRef: { current: null },
      db: {
        upsertChapter: vi.fn().mockResolvedValue(undefined),
        upsertCharacter: vi.fn().mockResolvedValue(undefined),
      },
    } as never;

    const { result } = renderHook(() => useChapterAnalysis(deps));

    await act(async () => {
      await result.current.handleAnalyze();
    });

    expect(vi.mocked(workerCall).mock.calls.map(([command]) => command)).toEqual([
      "analyze_chapter",
    ]);
    expect(vi.mocked(invoke)).toHaveBeenCalledWith("run_worker", {
      command: "_read_file",
      inputJson: JSON.stringify({ path: "/tmp/chapter.json" }),
    });
  });
});
