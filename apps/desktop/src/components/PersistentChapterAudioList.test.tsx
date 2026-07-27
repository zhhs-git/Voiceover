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
    expect(audio?.parentElement?.parentElement).toHaveClass("is-hidden");
  });
});
