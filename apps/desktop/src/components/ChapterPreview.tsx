import { useEffect, useMemo, useState } from "react";
import { readChapterText } from "../lib/platform";
import type { BookState } from "../types";

interface ChapterPreviewProps {
  book: BookState;
  onContinue: () => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function ChapterPreview({ book, onContinue }: ChapterPreviewProps) {
  const [previewChapterId, setPreviewChapterId] = useState<string | null>(
    book.chapters[0]?.id ?? null,
  );
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPreviewChapterId(book.chapters[0]?.id ?? null);
  }, [book.bookId]);

  useEffect(() => {
    if (previewChapterId && book.chapters.some((chapter) => chapter.id === previewChapterId)) {
      return;
    }
    setPreviewChapterId(book.chapters[0]?.id ?? null);
  }, [book.chapters, previewChapterId]);

  const activeChapter = useMemo(
    () =>
      book.chapters.find((chapter) => chapter.id === previewChapterId) ??
      book.chapters[0],
    [book.chapters, previewChapterId],
  );

  useEffect(() => {
    if (!activeChapter) {
      setText("");
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setText("");
    setError(null);
    setLoading(true);

    readChapterText(book.bookId, activeChapter.id, controller.signal)
      .then((chapterText) => {
        if (!controller.signal.aborted) setText(chapterText);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [book.bookId, activeChapter?.id]);

  const displayedLength = text.length > 0 ? text.length : activeChapter?.textLength ?? 0;

  return (
    <div className="chapter-preview">
      <header className="chapter-preview-header">
        <div>
          <p className="chapter-preview-eyebrow">章节文本</p>
          <h2>预览</h2>
        </div>
        <div className="chapter-preview-header-actions">
          <span className="chapter-preview-readonly">只读</span>
          <button
            className="btn-primary chapter-preview-continue"
            type="button"
            onClick={onContinue}
            disabled={book.chapters.length === 0}
          >
            进入分析
          </button>
        </div>
      </header>

      <div className="chapter-preview-body">
        <nav className="chapter-preview-list" aria-label="预览章节">
          <div className="chapter-preview-list-title">章节</div>
          {book.chapters.length === 0 ? (
            <div className="chapter-preview-empty">暂无章节</div>
          ) : (
            book.chapters.map((chapter) => (
              <button
                key={chapter.id}
                className={`chapter-preview-item ${
                  chapter.id === activeChapter?.id ? "active" : ""
                }`}
                type="button"
                aria-current={chapter.id === activeChapter?.id ? "page" : undefined}
                onClick={() => setPreviewChapterId(chapter.id)}
              >
                <span className="chapter-preview-item-title">{chapter.title}</span>
                <span className="chapter-preview-item-meta">
                  {chapter.textLength > 0 ? `${chapter.textLength} 字` : "文本"}
                </span>
              </button>
            ))
          )}
        </nav>

        <section className="chapter-preview-content" aria-live="polite">
          {activeChapter && (
            <header className="chapter-preview-content-header">
              <div>
                <h3>{activeChapter.title}</h3>
                <span>{displayedLength} 字</span>
              </div>
            </header>
          )}

          {loading && <div className="chapter-preview-state">正在读取章节文本…</div>}
          {!loading && error && (
            <div className="chapter-preview-state chapter-preview-error">
              读取失败：{error}
            </div>
          )}
          {!loading && !error && activeChapter && text.length === 0 && (
            <div className="chapter-preview-state">本章暂无文本内容。</div>
          )}
          {!loading && !error && text.length > 0 && (
            <pre className="chapter-preview-text">{text}</pre>
          )}
        </section>
      </div>
    </div>
  );
}
