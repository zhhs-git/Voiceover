import { describe, expect, test, vi } from "vitest";
import {
  cachedBookFromExtraction,
  extractionCachePath,
  writeExtractionCache,
} from "./importCache";

const book = {
  title: "Pride and Prejudice",
  bookId: "pride_and_prejudice",
  workDir: "/tmp/audiobook-generator/pride_and_prejudice",
  chapters: [
    {
      id: "chapter_001",
      title: "Chapter 1",
      textLength: 4773,
      textPath: "/tmp/audiobook-generator/pride_and_prejudice/chapters/chapter_001.txt",
    },
  ],
};

describe("importCache", () => {
  test("returns a cached book only when the source path matches", async () => {
    const readJson = vi.fn().mockResolvedValue({
      sourcePath: "/books/pride.epub",
      book,
    });

    const cached = await cachedBookFromExtraction({
      cachePath: "/tmp/audiobook-generator/pride_and_prejudice/book-extraction.json",
      sourcePath: "/books/pride.epub",
      readJson,
    });

    expect(cached).toEqual(book);
    expect(readJson).toHaveBeenCalledOnce();
  });

  test("ignores stale extraction cache for a different source path", async () => {
    const readJson = vi.fn().mockResolvedValue({
      sourcePath: "/books/other.epub",
      book,
    });

    const cached = await cachedBookFromExtraction({
      cachePath: "/tmp/audiobook-generator/pride_and_prejudice/book-extraction.json",
      sourcePath: "/books/pride.epub",
      readJson,
    });

    expect(cached).toBeNull();
  });

  test("writes extraction cache beside the work directory", async () => {
    const writeJson = vi.fn().mockResolvedValue(undefined);

    await writeExtractionCache({
      sourcePath: "/books/pride.epub",
      book,
      writeJson,
    });

    expect(writeJson).toHaveBeenCalledWith(
      extractionCachePath(book.workDir),
      {
        sourcePath: "/books/pride.epub",
        book,
      },
    );
  });
});
