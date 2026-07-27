import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { LibraryBook } from "../types";
import { LibraryView } from "./LibraryView";

const { listBooks, getChapters } = vi.hoisted(() => ({
  listBooks: vi.fn(),
  getChapters: vi.fn(),
}));

vi.mock("../state/store", () => ({
  createAudiobookStore: () => ({ listBooks, getChapters }),
}));

const book: LibraryBook = {
  id: "book_123",
  title: "Tiny Book",
  sourcePath: "/tmp/tiny.epub",
  workDir: "/tmp/audiobook-generator/books/book_123",
  importedAt: "2026-07-22T00:00:00Z",
};

describe("LibraryView", () => {
  beforeEach(() => {
    listBooks.mockResolvedValue([book]);
    getChapters.mockResolvedValue([]);
    vi.stubGlobal("confirm", vi.fn(() => true));
  });

  test("deletes a book from the library after confirmation", async () => {
    const onDeleteBook = vi.fn().mockResolvedValue(undefined);

    render(
      <LibraryView
        onImport={() => {}}
        onSelectBook={() => {}}
        onDeleteBook={onDeleteBook}
        importError={null}
      />,
    );

    const deleteButton = await screen.findByRole("button", {
      name: "删除 Tiny Book",
    });
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(onDeleteBook).toHaveBeenCalledWith(book);
      expect(screen.queryByText("Tiny Book")).not.toBeInTheDocument();
    });
  });

  test("does not delete a book when confirmation is cancelled", async () => {
    vi.stubGlobal("confirm", vi.fn(() => false));
    const onDeleteBook = vi.fn().mockResolvedValue(undefined);

    render(
      <LibraryView
        onImport={() => {}}
        onSelectBook={() => {}}
        onDeleteBook={onDeleteBook}
        importError={null}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: "删除 Tiny Book" }),
    );

    expect(onDeleteBook).not.toHaveBeenCalled();
    expect(screen.getByText("Tiny Book")).toBeInTheDocument();
  });
});
