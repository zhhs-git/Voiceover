import { describe, expect, test, vi } from "vitest";

const { invoke } = vi.hoisted(() => ({ invoke: vi.fn() }));

vi.mock("./platform", () => ({ invoke }));

import {
  batchErrorMessage,
  getActiveBatchGeneration,
  startBatchGeneration,
} from "./batchGeneration";

describe("batch generation API", () => {
  test("starts a selected chapter batch through the backend bridge", async () => {
    invoke.mockResolvedValueOnce({
      batchId: "batch_1",
      status: "queued",
      chapters: [],
    });

    await expect(
      startBatchGeneration({
        bookId: "book_1",
        chapterIds: ["chapter_001", "chapter_002"],
        cacheSegments: true,
      }),
    ).resolves.toMatchObject({ batchId: "batch_1", status: "queued" });

    expect(invoke).toHaveBeenCalledWith("batch_generation_start", {
      bookId: "book_1",
      chapterIds: ["chapter_001", "chapter_002"],
      cacheSegments: true,
    });
  });

  test("allows a page reload to restore an active batch", async () => {
    invoke.mockResolvedValueOnce(null);
    await expect(getActiveBatchGeneration("book_1")).resolves.toBeNull();
    expect(invoke).toHaveBeenCalledWith("batch_generation_active", { bookId: "book_1" });
  });

  test("extracts an understandable backend error", () => {
    expect(batchErrorMessage({ status: "failed", error: "timeout", chapters: [] })).toBe("timeout");
    expect(batchErrorMessage({ status: "failed", error: { message: "LLM timeout" }, chapters: [] })).toBe("LLM timeout");
  });
});
