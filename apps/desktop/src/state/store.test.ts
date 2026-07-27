// @vitest-environment node
import { beforeEach, describe, expect, test, vi } from "vitest";
import { createAudiobookStore } from "./store";

const invoke = vi.fn();

vi.mock("../lib/platform", () => ({
  invoke: (...args: unknown[]) => invoke(...args),
}));

describe("audiobook state store", () => {
  beforeEach(() => {
    invoke.mockReset();
  });

  test("delegates book and chapter persistence to the web bridge", async () => {
    const store = createAudiobookStore();

    await store.createBook({
      id: "book_123",
      title: "Tiny Book",
      sourcePath: "/tmp/tiny.epub",
      workDir: "/tmp/tiny",
    });
    await store.upsertChapter({
      id: "chapter_001",
      bookId: "book_123",
      title: "Chapter 1",
      status: "succeeded",
      scriptPath: "/tmp/tiny/scripts/chapter_001.json",
    });

    expect(invoke).toHaveBeenNthCalledWith(1, "db_create_book", {
      id: "book_123",
      title: "Tiny Book",
      sourcePath: "/tmp/tiny.epub",
      workDir: "/tmp/tiny",
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "db_upsert_chapter", {
      id: "chapter_001",
      bookId: "book_123",
      title: "Chapter 1",
      status: "succeeded",
      scriptPath: "/tmp/tiny/scripts/chapter_001.json",
    });
  });

  test("returns chapters with scripts from the web bridge", async () => {
    invoke.mockResolvedValueOnce([
      { id: "chapter_001", scriptPath: "/tmp/tiny/scripts/chapter_001.json" },
    ]);

    const store = createAudiobookStore();
    const chapters = await store.getChaptersWithScripts("book_123");

    expect(invoke).toHaveBeenCalledWith("db_get_chapters_with_scripts", {
      bookId: "book_123",
    });
    expect(chapters).toEqual([
      { id: "chapter_001", scriptPath: "/tmp/tiny/scripts/chapter_001.json" },
    ]);
  });

  test("deletes a book through the Tauri command", async () => {
    const store = createAudiobookStore();

    await store.deleteBook("book_123");

    expect(invoke).toHaveBeenCalledWith("db_delete_book", {
      bookId: "book_123",
    });
  });
});
