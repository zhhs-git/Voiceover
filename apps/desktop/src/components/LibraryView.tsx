import { useEffect, useState } from "react";
import type { LibraryBook } from "../types";
import { createAudiobookStore } from "../state/store";

const db = createAudiobookStore();

interface LibraryViewProps {
  onImport: () => void;
  onSelectBook: (book: LibraryBook) => void;
  onDeleteBook: (book: LibraryBook) => Promise<void>;
  importError: string | null;
}

function chapterProgressText(book: LibraryBook, chapters: Map<string, { total: number; generated: number }>): string {
  const info = chapters.get(book.id);
  if (!info) return "—";
  if (info.generated === 0) return `共 ${info.total} 章`;
  return `${info.generated} / ${info.total} 已生成`;
}

function progressPercent(book: LibraryBook, chapters: Map<string, { total: number; generated: number }>): number {
  const info = chapters.get(book.id);
  if (!info || info.total === 0) return 0;
  return Math.round((info.generated / info.total) * 100);
}

function formatDate(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("zh-CN", { month: "long", day: "numeric", year: "numeric" });
}

export function LibraryView({ onImport, onSelectBook, onDeleteBook, importError }: LibraryViewProps) {
  const [books, setBooks] = useState<LibraryBook[]>([]);
  const [chapterInfo, setChapterInfo] = useState<Map<string, { total: number; generated: number }>>(new Map());
  const [deletingBookId, setDeletingBookId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const list = await db.listBooks();
      if (cancelled) return;
      setBooks(list);

      const chapterResults = await Promise.all(list.map((b) => db.getChapters(b.id)));
      if (cancelled) return;
      const info = new Map<string, { total: number; generated: number }>();
      for (let i = 0; i < list.length; i++) {
        const chapters = chapterResults[i];
        const generated = chapters.filter((c) => c.status === "succeeded").length;
        info.set(list[i].id, { total: chapters.length, generated });
      }
      setChapterInfo(info);
    }
    load();
    return () => { cancelled = true; };
  }, []);

  async function handleDeleteBook(book: LibraryBook) {
    if (!window.confirm(`确定删除《${book.title}》及其所有生成文件？`)) return;
    setDeletingBookId(book.id);
    try {
      await onDeleteBook(book);
      setBooks((current) => current.filter((item) => item.id !== book.id));
      setChapterInfo((current) => {
        const next = new Map(current);
        next.delete(book.id);
        return next;
      });
    } catch {
    } finally {
      setDeletingBookId(null);
    }
  }

  if (books.length === 0) {
    return (
      <main className="library-view">
        <header className="library-topbar">
          <span className="library-wordmark">有声书工作台</span>
        </header>
        <div className="library-empty">
          <div className="empty-icon">📚</div>
          <h2>暂无书籍</h2>
          <p>导入 EPUB、PDF 或 TXT 文件，开始制作有声书。</p>
          <button className="btn-primary library-import-btn" onClick={onImport}>
            + 导入书籍
          </button>
          {importError && <p className="error-text">{importError}</p>}
        </div>
      </main>
    );
  }

  return (
    <main className="library-view">
      <header className="library-topbar">
        <span className="library-wordmark">有声书工作台</span>
        <button className="btn-primary" style={{ width: "auto", padding: "8px 16px", fontSize: "12px" }} onClick={onImport}>
          + 导入
        </button>
      </header>

      {importError && <p className="error-text" style={{ margin: "12px 28px 0" }}>{importError}</p>}

      <div className="library-header">
        <h1>书库</h1>
        <span className="chapter-count-badge">{books.length} 本</span>
      </div>

      <div className="library-grid">
        {books.map((book) => {
          const pct = progressPercent(book, chapterInfo);
          return (
            <div
              key={book.id}
              className="library-card"
            >
              <div className="card-cover-area">📖</div>
              <div className="card-body">
                <button
                  className="card-open-button"
                  type="button"
                  onClick={() => onSelectBook(book)}
                >
                  <span className="card-title">{book.title}</span>
                </button>
                <div className="card-progress">
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="progress-text">
                    {chapterProgressText(book, chapterInfo)}
                  </span>
                </div>
                <div className="card-date">{formatDate(book.importedAt)}</div>
                <button
                  className="card-delete-button"
                  type="button"
                  aria-label={`删除 ${book.title}`}
                  onClick={() => void handleDeleteBook(book)}
                  disabled={deletingBookId === book.id}
                >
                  {deletingBookId === book.id ? "删除中…" : "删除"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </main>
  );
}
