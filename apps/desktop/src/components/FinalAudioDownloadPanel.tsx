import { useEffect, useMemo, useRef, useState } from "react";

import {
  downloadFinalAudioArchive,
  type FinalAudioExportFormat,
} from "../lib/platform";
import type { ChapterMeta, ChapterWorkflowStatus } from "../types";

const MP3_BITRATES = [128, 192, 256, 320] as const;

interface FinalAudioDownloadPanelProps {
  bookId: string;
  chapters: ChapterMeta[];
  chapterMixedAudioPaths: Record<string, string>;
  generationWorkflows: Record<string, ChapterWorkflowStatus>;
}

function unavailableStatus(workflow: ChapterWorkflowStatus | undefined): string {
  if (workflow?.status === "running") return "生成中";
  if (workflow?.status === "failed") return "生成失败";
  if (workflow?.status === "succeeded") return "最终音频缺失";
  return "未完成";
}

export function FinalAudioDownloadPanel({
  bookId,
  chapters,
  chapterMixedAudioPaths,
  generationWorkflows,
}: FinalAudioDownloadPanelProps) {
  const [selectedChapterIds, setSelectedChapterIds] = useState<Set<string>>(() => new Set());
  const [format, setFormat] = useState<FinalAudioExportFormat>("mp3");
  const [bitrateKbps, setBitrateKbps] = useState<number>(192);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const initializedBookIdRef = useRef<string | null>(null);
  const hasAppliedDefaultSelectionRef = useRef(false);

  const availableChapterIds = useMemo(
    () => chapters
      .filter((chapter) => Boolean(chapterMixedAudioPaths[chapter.id]))
      .map((chapter) => chapter.id),
    [chapters, chapterMixedAudioPaths],
  );
  const availableChapterIdSet = useMemo(() => new Set(availableChapterIds), [availableChapterIds]);
  const availabilityKey = availableChapterIds.join("\u0000");

  useEffect(() => {
    if (initializedBookIdRef.current !== bookId) {
      initializedBookIdRef.current = bookId;
      hasAppliedDefaultSelectionRef.current = false;
      setSelectedChapterIds(new Set());
    }
    if (availableChapterIds.length === 0) {
      setSelectedChapterIds((current) => new Set([...current].filter((id) => availableChapterIdSet.has(id))));
      return;
    }
    if (!hasAppliedDefaultSelectionRef.current) {
      hasAppliedDefaultSelectionRef.current = true;
      setSelectedChapterIds(new Set(availableChapterIds));
      return;
    }
    setSelectedChapterIds((current) => new Set([...current].filter((id) => availableChapterIdSet.has(id))));
  }, [availableChapterIds, availableChapterIdSet, availabilityKey, bookId]);

  const selectedAvailableChapterIds = chapters
    .map((chapter) => chapter.id)
    .filter((chapterId) => availableChapterIdSet.has(chapterId) && selectedChapterIds.has(chapterId));
  const allAvailableSelected =
    availableChapterIds.length > 0 && selectedAvailableChapterIds.length === availableChapterIds.length;

  function toggleChapter(chapterId: string) {
    setResult(null);
    setError(null);
    setSelectedChapterIds((current) => {
      const next = new Set(current);
      if (next.has(chapterId)) next.delete(chapterId);
      else next.add(chapterId);
      return next;
    });
  }

  function toggleAll() {
    setResult(null);
    setError(null);
    setSelectedChapterIds(allAvailableSelected ? new Set() : new Set(availableChapterIds));
  }

  async function handleDownload() {
    if (isDownloading || selectedAvailableChapterIds.length === 0) return;
    setIsDownloading(true);
    setError(null);
    setResult(null);
    try {
      const response = await downloadFinalAudioArchive({
        bookId,
        chapterIds: selectedAvailableChapterIds,
        format,
        ...(format === "mp3" ? { bitrateKbps } : {}),
      });
      const skipped = response.skippedCount > 0 ? `，跳过 ${response.skippedCount} 章` : "";
      setResult(`已开始下载 ${response.chapterCount} 章${skipped}。`);
    } catch (downloadError) {
      setError(`下载失败：${String(downloadError)}`);
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <section className="final-audio-download-panel" aria-label="批量下载最终配音">
      <header className="final-audio-download-header">
        <div>
          <h2>下载最终配音</h2>
          <p>{availableChapterIds.length} 章可下载，已选择 {selectedAvailableChapterIds.length} 章</p>
        </div>
        <button
          type="button"
          className="btn-secondary btn-sm"
          onClick={toggleAll}
          disabled={availableChapterIds.length === 0 || isDownloading}
        >
          {allAvailableSelected ? "取消全选" : "全选可下载章节"}
        </button>
      </header>

      <div className="final-audio-download-controls">
        <div className="final-audio-format-control" role="group" aria-label="下载格式">
          {(["mp3", "wav"] as const).map((value) => (
            <button
              type="button"
              key={value}
              className={`final-audio-format-option ${format === value ? "active" : ""}`}
              aria-pressed={format === value}
              onClick={() => {
                setFormat(value);
                setResult(null);
                setError(null);
              }}
              disabled={isDownloading}
            >
              {value.toUpperCase()}
            </button>
          ))}
        </div>
        {format === "mp3" && (
          <label className="final-audio-bitrate-control">
            <span>码率</span>
            <select
              aria-label="MP3 码率"
              value={bitrateKbps}
              onChange={(event) => setBitrateKbps(Number(event.target.value))}
              disabled={isDownloading}
            >
              {MP3_BITRATES.map((value) => (
                <option key={value} value={value}>{value} kbps</option>
              ))}
            </select>
          </label>
        )}
        <button
          type="button"
          className="btn-primary final-audio-download-button"
          onClick={() => void handleDownload()}
          disabled={isDownloading || selectedAvailableChapterIds.length === 0}
        >
          {isDownloading ? "正在打包…" : `下载 ${format.toUpperCase()} ZIP`}
        </button>
      </div>

      {error && <p className="final-audio-download-error" role="alert">{error}</p>}
      {result && <p className="final-audio-download-result">{result}</p>}

      <div className="final-audio-download-list" role="list">
        {chapters.map((chapter, index) => {
          const available = availableChapterIdSet.has(chapter.id);
          return (
            <label
              className={`final-audio-download-row ${available ? "available" : "unavailable"}`}
              key={chapter.id}
              role="listitem"
            >
              <input
                type="checkbox"
                aria-label={`选择《${chapter.title}》`}
                checked={selectedChapterIds.has(chapter.id)}
                onChange={() => toggleChapter(chapter.id)}
                disabled={!available || isDownloading}
              />
              <span className="final-audio-download-position">{index + 1}</span>
              <span className="final-audio-download-title">{chapter.title}</span>
              <span className={`final-audio-download-status ${available ? "ready" : "pending"}`}>
                {available ? "可下载" : unavailableStatus(generationWorkflows[chapter.id])}
              </span>
            </label>
          );
        })}
      </div>
    </section>
  );
}
