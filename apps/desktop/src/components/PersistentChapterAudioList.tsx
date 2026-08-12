import { convertFileSrc } from "../lib/platform";
import type { ChapterMeta } from "../types";

interface PersistentChapterAudioListProps {
  chapters: ChapterMeta[];
  chapterAudioPaths: Record<string, string>;
  chapterMixedAudioPaths?: Record<string, string>;
  activeChapterId?: string;
  visible: boolean;
  onDownload: (chapterId: string, chapterTitle: string) => void;
  onDownloadMixed?: (chapterId: string, chapterTitle: string, path: string) => void;
  onRegenerate: (chapter: ChapterMeta) => void;
  onRegenerateFinal?: (chapter: ChapterMeta) => void;
}

export function PersistentChapterAudioList({
  chapters,
  chapterAudioPaths,
  chapterMixedAudioPaths = {},
  activeChapterId,
  visible,
  onDownload,
  onDownloadMixed,
  onRegenerate,
  onRegenerateFinal,
}: PersistentChapterAudioListProps) {
  const generatedChapters = chapters.filter(
    (chapter) =>
      (!activeChapterId || chapter.id === activeChapterId) &&
      (chapterAudioPaths[chapter.id] || chapterMixedAudioPaths[chapter.id]),
  );
  if (generatedChapters.length === 0) return null;

  const tabIndex = visible ? 0 : -1;

  return (
    <div
      className={`tab-panel chapter-audio-persistence ${visible ? "is-visible" : "is-hidden"}`}
      aria-hidden={!visible}
    >
      {generatedChapters.map((chapter) => {
        const voicePath = chapterAudioPaths[chapter.id];
        const mixedPath = chapterMixedAudioPaths[chapter.id];
        return (
          <div key={chapter.id} className="chapter-audio-pair">
            {voicePath && (
              <div className="audio-row">
                <span>{chapter.title} · 原章节配音</span>
                <audio
                  controls
                  tabIndex={tabIndex}
                  aria-label={`原章节配音：${chapter.title}`}
                  src={convertFileSrc(voicePath)}
                />
                <button
                  className="btn-secondary"
                  tabIndex={tabIndex}
                  onClick={() => onDownload(chapter.id, chapter.title)}
                  title="下载为 MP3"
                >
                  下载 MP3
                </button>
                <button
                  className="btn-secondary"
                  tabIndex={tabIndex}
                  onClick={() => onRegenerate(chapter)}
                >
                  重新生成原配音
                </button>
              </div>
            )}
            {mixedPath && (
              <div className="audio-row">
                <span>{chapter.title} · 最终配音（混音）</span>
                <audio
                  controls
                  tabIndex={tabIndex}
                  aria-label={`最终配音（混音）：${chapter.title}`}
                  src={convertFileSrc(mixedPath)}
                />
                <button
                  className="btn-secondary"
                  tabIndex={tabIndex}
                  onClick={() => onDownloadMixed?.(chapter.id, chapter.title, mixedPath)}
                  title="下载为 MP3"
                >
                  下载 MP3
                </button>
                {onRegenerateFinal && (
                  <button
                    className="btn-secondary"
                    tabIndex={tabIndex}
                    onClick={() => onRegenerateFinal(chapter)}
                  >
                    重新生成最终配音
                  </button>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
