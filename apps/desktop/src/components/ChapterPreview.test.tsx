import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { BookState } from "../types";
import { ChapterPreview } from "./ChapterPreview";

const { readChapterText } = vi.hoisted(() => ({
  readChapterText: vi.fn(),
}));

vi.mock("../lib/platform", () => ({ readChapterText }));

const book: BookState = {
  title: "测试书",
  bookId: "book_123",
  workDir: "/data/books/book_123",
  chapters: [
    {
      id: "chapter_001",
      title: "第一章",
      textLength: 8,
      textPath: "/data/books/book_123/chapters/chapter_001.txt",
    },
    {
      id: "chapter_002",
      title: "第二章",
      textLength: 8,
      textPath: "/data/books/book_123/chapters/chapter_002.txt",
    },
  ],
};

describe("ChapterPreview", () => {
  beforeEach(() => {
    readChapterText.mockImplementation((_bookId: string, chapterId: string) =>
      Promise.resolve(chapterId === "chapter_001" ? "第一章正文\n第二行" : "第二章正文"),
    );
  });

  test("loads the first chapter and switches preview text independently", async () => {
    render(<ChapterPreview book={book} onContinue={vi.fn()} />);

    expect(await screen.findByText(/第一章正文/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /第二章/ }));

    expect(await screen.findByText("第二章正文")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText(/第一章正文/)).not.toBeInTheDocument();
    });
    expect(readChapterText).toHaveBeenCalledWith(
      "book_123",
      "chapter_002",
      expect.any(AbortSignal),
    );
  });

  test("continues to analysis without changing the preview selection state", async () => {
    const onContinue = vi.fn();
    render(<ChapterPreview book={book} onContinue={onContinue} />);

    await screen.findByText(/第一章正文/);
    fireEvent.click(screen.getByRole("button", { name: "进入分析" }));

    expect(onContinue).toHaveBeenCalledOnce();
  });
});
