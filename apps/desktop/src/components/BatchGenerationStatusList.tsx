import { useEffect, useState } from "react";

import type {
  BatchGenerationChapter,
  BatchGenerationResponse,
  BatchGenerationStatus,
} from "../lib/batchGeneration";
import {
  batchChapterDisplayStage,
  isBatchChapterMiMoInProgress,
  isBatchChapterStageInProgress,
  isBatchChapterWaitingForMiMo,
} from "../lib/batchGeneration";

const BATCH_STATUS_LABELS: Record<BatchGenerationStatus, string> = {
  queued: "等待后台队列",
  running: "进行中",
  succeeded: "全部完成",
  completed_with_errors: "已完成（有失败）",
  cancelled: "已停止",
  failed: "任务失败",
};

const CHAPTER_STATUS_LABELS: Record<BatchGenerationChapter["status"], string> = {
  queued: "排队中",
  running: "进行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已停止",
};

const STAGE_LABELS: Record<string, string> = {
  voice: "原章节配音",
  transcript: "Whisper 转录",
  audio_plan: "背景音/音效规划",
  stable_audio: "Stable Audio 背景音/音效",
  mix: "最终混音",
};

function stageLabel(stage: string | null | undefined): string | null {
  if (!stage) return null;
  return STAGE_LABELS[stage] ?? stage;
}

function chapterStatusLabel(chapter: BatchGenerationChapter): string {
  if (isBatchChapterWaitingForMiMo(chapter)) return "等待 MiMo";
  if (isBatchChapterMiMoInProgress(chapter)) return "MiMo 配音中";
  return CHAPTER_STATUS_LABELS[chapter.status];
}

function chapterDetail(
  chapter: BatchGenerationChapter,
  mimoCooldownSeconds: number | null | undefined,
): string {
  if (chapter.status === "failed") return chapter.error || "生成失败";
  if (isBatchChapterWaitingForMiMo(chapter)) {
    if (typeof mimoCooldownSeconds === "number" && Number.isFinite(mimoCooldownSeconds) && mimoCooldownSeconds > 0) {
      return `MiMo 限流冷却中，约${formatBatchElapsed(Math.ceil(mimoCooldownSeconds))}后继续`;
    }
    return "等待 MiMo 串行配音";
  }
  if (isBatchChapterMiMoInProgress(chapter)) return "正在原章节配音（MiMo 串行）";
  const stage = stageLabel(batchChapterDisplayStage(chapter));
  if (isBatchChapterStageInProgress(chapter) && stage) return `正在${stage}`;
  return chapter.stageState === "ready" && stage ? `等待${stage}` : "";
}

function chapterTimingTitle(chapter: BatchGenerationChapter): string | undefined {
  const timings = chapter.stageTimings;
  if (!timings || Object.keys(timings).length === 0) return undefined;
  return Object.entries(timings)
    .filter(([, seconds]) => typeof seconds === "number" && Number.isFinite(seconds))
    .map(([stage, seconds]) => `${stageLabel(stage) ?? stage}：${formatBatchElapsed(seconds)}`)
    .join("；");
}

function isActiveBatch(status: BatchGenerationStatus): boolean {
  return status === "queued" || status === "running";
}

/** Format a duration measured from task submission, including queue wait time. */
export function formatBatchElapsed(seconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours} 小时 ${String(minutes).padStart(2, "0")} 分 ${String(remainingSeconds).padStart(2, "0")} 秒`;
  }
  if (minutes > 0) {
    return `${minutes} 分 ${String(remainingSeconds).padStart(2, "0")} 秒`;
  }
  return `${remainingSeconds} 秒`;
}

export function batchElapsedSeconds(
  batch: Pick<BatchGenerationResponse, "createdAt" | "completedAt">,
  nowMilliseconds: number,
): number | null {
  if (typeof batch.createdAt !== "number" || !Number.isFinite(batch.createdAt)) {
    return null;
  }
  const completedAt = batch.completedAt;
  const endSeconds =
    typeof completedAt === "number" && Number.isFinite(completedAt)
      ? completedAt
      : nowMilliseconds / 1000;
  return Math.max(0, Math.floor(endSeconds - batch.createdAt));
}

interface BatchGenerationStatusListProps {
  batch: BatchGenerationResponse | null;
}

/**
 * The queue is authoritative for bulk work. It is deliberately rendered from
 * the server response instead of opening a workflow-file polling loop per
 * chapter, which would not scale to long books.
 */
export function BatchGenerationStatusList({ batch }: BatchGenerationStatusListProps) {
  const [nowMilliseconds, setNowMilliseconds] = useState(() => Date.now());
  const active = batch ? isActiveBatch(batch.status) : false;

  useEffect(() => {
    setNowMilliseconds(Date.now());
    if (!active) return undefined;
    const intervalId = window.setInterval(() => setNowMilliseconds(Date.now()), 1000);
    return () => window.clearInterval(intervalId);
  }, [active, batch?.batchId, batch?.createdAt, batch?.completedAt]);

  if (!batch || batch.chapters.length === 0) return null;

  const completed = batch.completedCount ?? batch.chapters.filter((chapter) =>
    chapter.status === "succeeded" || chapter.status === "failed" || chapter.status === "cancelled",
  ).length;
  const elapsedSeconds = batchElapsedSeconds(batch, nowMilliseconds);

  return (
    <section className="batch-generation-status" aria-label="批量生成队列状态">
      <div className="batch-generation-status-header">
        <div>
          <h3>批量生成队列</h3>
          <p className="batch-generation-meta">
            <span>{completed} / {batch.totalCount ?? batch.chapters.length} 章已处理</span>
            {elapsedSeconds !== null && (
              <span title="从提交批量任务起计算，包含等待后台队列时间">
                总耗时：{formatBatchElapsed(elapsedSeconds)}
              </span>
            )}
          </p>
        </div>
        <span className={`workflow-summary-status batch-status-${batch.status}`}>
          {BATCH_STATUS_LABELS[batch.status]}
        </span>
      </div>
      <div className="batch-generation-chapter-list" role="list">
        {batch.chapters.map((chapter) => {
          const detail = chapterDetail(chapter, batch.mimoCooldownSeconds);
          const duration = typeof chapter.durationSeconds === "number" && Number.isFinite(chapter.durationSeconds)
            ? formatBatchElapsed(chapter.durationSeconds)
            : null;
          const timingTitle = chapterTimingTitle(chapter);
          return (
            <div className={`batch-generation-chapter batch-chapter-${chapter.status}`} key={chapter.chapterId} role="listitem">
              <span className="batch-generation-position">{chapter.position + 1}</span>
              <span className="batch-generation-title">{chapter.title}</span>
              <span className={`batch-generation-chapter-status batch-chapter-${chapter.status}`}>
                {chapterStatusLabel(chapter)}
              </span>
              {duration && (
                <span className="batch-generation-duration" title={timingTitle}>
                  耗时 {duration}
                </span>
              )}
              {detail && <span className="batch-generation-detail" title={detail}>{detail}</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
