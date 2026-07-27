import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { StepDone } from "./StepDone";

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: (path: string) => `asset://localhost/${path}`,
}));

const book = {
  title: "Pride and Prejudice",
  bookId: "pride",
  workDir: "/tmp/pride",
  chapters: [
    { id: "chapter_001", title: "Chapter 1", textLength: 1200, textPath: "/tmp/ch1.txt" },
    { id: "chapter_002", title: "Chapter 2", textLength: 900, textPath: "/tmp/ch2.txt" },
  ],
};

describe("StepDone", () => {
  test("offers chapter and all-chapter regeneration", () => {
    const onRegenerateChapter = vi.fn();
    const onRegenerateAll = vi.fn();

    render(
      <StepDone
        book={book}
        chapterAudioPaths={{
          chapter_001: "/tmp/chapter_001.wav",
          chapter_002: "/tmp/chapter_002.wav",
        }}
        analysis={null}
        savedMessage={null}
        isBusy={false}
        isGenerating={false}
        analyzeProgress=""
        progressDetail={[]}
        progress={0}
        onSaveChapter={() => {}}
        onRegenerateChapter={onRegenerateChapter}
        onRegenerateAll={onRegenerateAll}
      />,
    );

    screen.getByRole("button", { name: /全部重新生成/i }).click();
    expect(onRegenerateAll).toHaveBeenCalledOnce();

    screen.getAllByRole("button", { name: /重新生成章节/i })[0].click();
    expect(onRegenerateChapter).toHaveBeenCalledWith(book.chapters[0]);
  });
});
