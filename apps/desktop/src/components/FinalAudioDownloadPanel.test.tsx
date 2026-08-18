import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import type { ChapterMeta, ChapterWorkflowStatus } from "../types";
import { downloadFinalAudioArchive } from "../lib/platform";
import { FinalAudioDownloadPanel } from "./FinalAudioDownloadPanel";

vi.mock("../lib/platform", () => ({
  downloadFinalAudioArchive: vi.fn(),
}));

afterEach(() => {
  vi.clearAllMocks();
});

const chapters: ChapterMeta[] = [
  {
    id: "chapter_001",
    title: "第一章",
    textLength: 100,
    textPath: "/books/book_123/chapters/chapter_001.txt",
  },
  {
    id: "chapter_002",
    title: "第二章",
    textLength: 120,
    textPath: "/books/book_123/chapters/chapter_002.txt",
  },
  {
    id: "chapter_003",
    title: "第三章",
    textLength: 80,
    textPath: "/books/book_123/chapters/chapter_003.txt",
  },
];

const generationWorkflows: Record<string, ChapterWorkflowStatus> = {
  chapter_003: {
    chapterId: "chapter_003",
    kind: "generation",
    status: "running",
    steps: [],
  },
};

function renderPanel() {
  return render(
    <FinalAudioDownloadPanel
      bookId="book_123"
      chapters={chapters}
      chapterMixedAudioPaths={{
        chapter_001: "/books/book_123/audio/chapter_001_mixed.wav",
        chapter_002: "/books/book_123/audio/chapter_002_mixed.wav",
      }}
      generationWorkflows={generationWorkflows}
    />,
  );
}

describe("FinalAudioDownloadPanel", () => {
  test("selects completed final mixes by default and excludes unavailable chapters", async () => {
    renderPanel();

    const first = screen.getByLabelText("选择《第一章》") as HTMLInputElement;
    const second = screen.getByLabelText("选择《第二章》") as HTMLInputElement;
    const third = screen.getByLabelText("选择《第三章》") as HTMLInputElement;
    await waitFor(() => {
      expect(first.checked).toBe(true);
      expect(second.checked).toBe(true);
    });
    expect(third.checked).toBe(false);
    expect(third.disabled).toBe(true);
    expect(screen.getByText("生成中")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "下载 MP3 ZIP" })).toBeEnabled();
  });

  test("submits the remaining selected chapters with the chosen fixed MP3 bitrate", async () => {
    vi.mocked(downloadFinalAudioArchive).mockResolvedValue({
      filename: "book-final-audio-mp3-256kbps.zip",
      chapterCount: 1,
      skippedCount: 0,
    });
    renderPanel();
    await waitFor(() => expect(screen.getByLabelText("选择《第一章》")).toBeChecked());

    fireEvent.click(screen.getByLabelText("选择《第二章》"));
    fireEvent.change(screen.getByLabelText("MP3 码率"), { target: { value: "256" } });
    fireEvent.click(screen.getByRole("button", { name: "下载 MP3 ZIP" }));

    await waitFor(() => {
      expect(downloadFinalAudioArchive).toHaveBeenCalledWith({
        bookId: "book_123",
        chapterIds: ["chapter_001"],
        format: "mp3",
        bitrateKbps: 256,
      });
    });
    expect(screen.getByText("已开始下载 1 章。")).toBeInTheDocument();
  });

  test("hides bitrate controls and omits bitrate when exporting WAV", async () => {
    vi.mocked(downloadFinalAudioArchive).mockResolvedValue({
      filename: "book-final-audio-wav.zip",
      chapterCount: 2,
      skippedCount: 0,
    });
    renderPanel();
    await waitFor(() => expect(screen.getByLabelText("选择《第一章》")).toBeChecked());

    fireEvent.click(screen.getByRole("button", { name: "WAV" }));
    expect(screen.queryByLabelText("MP3 码率")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "下载 WAV ZIP" }));

    await waitFor(() => {
      expect(downloadFinalAudioArchive).toHaveBeenCalledWith({
        bookId: "book_123",
        chapterIds: ["chapter_001", "chapter_002"],
        format: "wav",
      });
    });
  });
});
