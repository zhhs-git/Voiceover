import { convertFileSrc } from "../lib/platform";
import type { ChapterMeta } from "../types";

interface PersistentChapterAudioListProps {
  chapters: ChapterMeta[];
  chapterAudioPaths: Record<string, string>;
  visible: boolean;
  onDownload: (chapterId: string, chapterTitle: string) => void;
  onRegenerate: (chapter: ChapterMeta) => void;
}

export function PersistentChapterAudioList({
  chapters,
  chapterAudioPaths,
  visible,
  onDownload,
  onRegenerate,
}: PersistentChapterAudioListProps) {
  const generatedChapters = chapters.filter((chapter) => chapterAudioPaths[chapter.id]);
  if (generatedChapters.length === 0) return null;

  const tabIndex = visible ? 0 : -1;

  return (
    <div
      className={`tab-panel chapter-audio-persistence ${visible ? "is-visible" : "is-hidden"}`}
      aria-hidden={!visible}
    >
      {generatedChapters.map((chapter) => (
        <div key={chapter.id} className="audio-row">
          <span>{chapter.title}</span>
          <audio
            controls
            tabIndex={tabIndex}
            aria-label={`章节音频：${chapter.title}`}
            src={convertFileSrc(chapterAudioPaths[chapter.id])}
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
            重新生成
          </button>
        </div>
      ))}
    </div>
  );
}
