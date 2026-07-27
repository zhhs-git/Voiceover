import { useCallback } from "react";
import { importBook, invoke, selectBookFile } from "./lib/platform";
import type { BookState, LibraryBook } from "./types";
import { LibraryView } from "./components/LibraryView";
import { BookDetailView } from "./components/BookDetailView";
import { createAudiobookStore } from "./state/store";
import { useAppStore } from "./state/appStore";
import {
  cachedBookFromExtraction,
  extractionCachePath,
  writeExtractionCache,
} from "./lib/importCache";

const db = createAudiobookStore();

export function App() {
  const {
    view,
    activeBook,
    activeSourcePath,
    importError,
    navigateToLibrary,
    navigateToBook,
    setImportError,
  } = useAppStore();

  const handleImport = useCallback(async () => {
    const file = await selectBookFile();
    if (!file) return;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !["epub", "pdf", "txt"].includes(extension)) {
      setImportError("不支持的文件类型，请选择 EPUB、PDF 或 TXT 文件。");
      return;
    }
    setImportError(null);
    try {
      const result = await importBook(file);
      if (result.status !== "succeeded") {
        throw new Error((result.error as any)?.message ?? "书籍提取失败");
      }
      const artifact = (
        result.artifacts as unknown as Array<{
          metadata: {
            title: string;
            bookId: string;
            workDir: string;
            sourcePath: string;
            chapters: { id: string; title: string; textLength: number; textPath: string }[];
          };
        }>
      )[0];
      const extracted: BookState = {
        title: artifact.metadata.title,
        bookId: artifact.metadata.bookId,
        workDir: artifact.metadata.workDir,
        chapters: artifact.metadata.chapters,
      };
      navigateToBook(extracted, artifact.metadata.sourcePath);
    } catch (err) {
      setImportError(`导入失败：${String(err)}`);
    }
  }, [navigateToBook, setImportError]);

  const handleDeleteBook = useCallback(async (book: LibraryBook) => {
    setImportError(null);
    try {
      await db.deleteBook(book.id);
    } catch (err) {
      setImportError(`删除《${book.title}》失败：${String(err)}`);
      throw err;
    }
  }, [setImportError]);

  if (view.page === "library") {
    return (
      <LibraryView
        onImport={handleImport}
        onDeleteBook={handleDeleteBook}
        onSelectBook={async (libBook: LibraryBook) => {
          const cache = await cachedBookFromExtraction({
            cachePath: extractionCachePath(libBook.workDir),
            sourcePath: libBook.sourcePath,
            readJson: async (p) =>
              await invoke("run_worker", {
                command: "_read_file",
                inputJson: JSON.stringify({ path: p }),
              }),
          });
          if (cache) {
            navigateToBook(cache, libBook.sourcePath);
            return;
          }
          const chapters = await db.getChapters(libBook.id);
          navigateToBook({
            title: libBook.title,
            bookId: libBook.id,
            workDir: libBook.workDir,
            chapters: chapters.map((c) => ({
              id: c.id,
              title: c.title,
              textLength: 0,
              textPath: `${libBook.workDir}/chapters/${c.id}.txt`,
            })),
          }, libBook.sourcePath);
        }}
        importError={importError}
      />
    );
  }

  if (activeBook && view.page === "bookDetail") {
    const libBook: LibraryBook = {
      id: activeBook.bookId,
      title: activeBook.title,
      sourcePath: activeSourcePath,
      workDir: activeBook.workDir,
      importedAt: null,
    };
    return (
      <BookDetailView
        libraryBook={libBook}
        book={activeBook}
        sourcePath={activeSourcePath}
        onBack={navigateToLibrary}
        onBookUpdate={(b) => navigateToBook(b, activeSourcePath)}
      />
    );
  }

  return null;
}
