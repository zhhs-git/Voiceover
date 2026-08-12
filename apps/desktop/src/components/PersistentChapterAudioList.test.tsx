import { render } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import type { ChapterMeta } from "../types";
import { PersistentChapterAudioList } from "./PersistentChapterAudioList";

vi.mock("../lib/platform", () => ({
  convertFileSrc: (path: string) => `/api/files?path=${encodeURIComponent(path)}`,
}));

const chapter: ChapterMeta = {
  id: "chapter_001",
  title: "第一章",
  textLength: 100,
  textPath: "/books/book_123/chapters/chapter_001.txt",
};

const secondChapter: ChapterMeta = {
  id: "chapter_002",
  title: "第二章",
  textLength: 120,
  textPath: "/books/book_123/chapters/chapter_002.txt",
};

describe("PersistentChapterAudioList", () => {
  test("keeps the same audio element mounted when the tab becomes hidden", () => {
    const { container, rerender } = render(
      <PersistentChapterAudioList
        chapters={[chapter]}
        chapterAudioPaths={{ chapter_001: "/books/book_123/audio/chapter_001.wav" }}
        visible
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio%2Fchapter_001.wav",
    );

    rerender(
      <PersistentChapterAudioList
        chapters={[chapter]}
        chapterAudioPaths={{ chapter_001: "/books/book_123/audio/chapter_001.wav" }}
        visible={false}
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    expect(container.querySelector("audio")).toBe(audio);
    expect(audio?.closest(".tab-panel")).toHaveClass("is-hidden");
  });

  test("renders original and final audio as separate tracks", () => {
    const { getByLabelText } = render(
      <PersistentChapterAudioList
        chapters={[chapter]}
        chapterAudioPaths={{ chapter_001: "/books/book_123/audio/chapter_001.wav" }}
        chapterMixedAudioPaths={{ chapter_001: "/books/book_123/audio/chapter_001_mixed.wav" }}
        visible
        onDownload={vi.fn()}
        onDownloadMixed={vi.fn()}
        onRegenerate={vi.fn()}
        onRegenerateFinal={vi.fn()}
      />,
    );

    expect(getByLabelText("原章节配音：第一章")).toBeInTheDocument();
    expect(getByLabelText("最终配音（混音）：第一章")).toBeInTheDocument();
  });

  test("only renders the selected listening chapter", () => {
    const { rerender } = render(
      <PersistentChapterAudioList
        chapters={[chapter, secondChapter]}
        chapterAudioPaths={{
          chapter_001: "/books/book_123/audio/chapter_001.wav",
          chapter_002: "/books/book_123/audio/chapter_002.wav",
        }}
        activeChapterId="chapter_001"
        visible
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    expect(document.querySelector("audio")).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio%2Fchapter_001.wav",
    );
    expect(document.body).toHaveTextContent("第一章 · 原章节配音");
    expect(document.body).not.toHaveTextContent("第二章 · 原章节配音");

    rerender(
      <PersistentChapterAudioList
        chapters={[chapter, secondChapter]}
        chapterAudioPaths={{
          chapter_001: "/books/book_123/audio/chapter_001.wav",
          chapter_002: "/books/book_123/audio/chapter_002.wav",
        }}
        activeChapterId="chapter_002"
        visible
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    expect(document.querySelector("audio")).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio%2Fchapter_002.wav",
    );
    expect(document.body).toHaveTextContent("第二章 · 原章节配音");
    expect(document.body).not.toHaveTextContent("第一章 · 原章节配音");
  });
});
