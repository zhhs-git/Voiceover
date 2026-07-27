import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { App } from "./App";

const selectBookFile = vi.fn().mockResolvedValue(null);
vi.mock("./lib/platform", () => ({
  invoke: vi.fn().mockResolvedValue(null),
  importBook: vi.fn(),
  selectBookFile: (...args: unknown[]) => selectBookFile(...args),
}));
vi.mock("./state/store", () => ({
  createAudiobookStore: () => ({
    listBooks: vi.fn().mockResolvedValue([]),
    getBook: vi.fn().mockResolvedValue(null),
    getChapters: vi.fn().mockResolvedValue([]),
    createBook: vi.fn().mockResolvedValue(null),
    upsertChapter: vi.fn().mockResolvedValue(null),
    getChaptersWithScripts: vi.fn().mockResolvedValue([]),
    upsertCharacter: vi.fn().mockResolvedValue(null),
    getCharacters: vi.fn().mockResolvedValue([]),
    deleteBook: vi.fn().mockResolvedValue(null),
  }),
}));

describe("App", () => {
  it("renders the library view", async () => {
    render(<App />);
    expect(screen.getByText("暂无书籍")).toBeDefined();
  });

  it("opens the browser file picker", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "+ 导入书籍" }));

    await waitFor(() => {
      expect(selectBookFile).toHaveBeenCalledOnce();
    });
  });
});
