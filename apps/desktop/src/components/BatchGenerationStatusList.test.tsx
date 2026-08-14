import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BatchGenerationStatusList,
  batchElapsedSeconds,
  formatBatchElapsed,
} from "./BatchGenerationStatusList";

describe("BatchGenerationStatusList", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows durable queue states and the current worker stage", () => {
    render(
      <BatchGenerationStatusList
        batch={{
          batchId: "batch_1",
          status: "running",
          totalCount: 3,
          completedCount: 1,
          chapters: [
            { chapterId: "chapter_001", title: "第一章", position: 0, status: "succeeded" },
            { chapterId: "chapter_002", title: "第二章", position: 1, status: "running", currentStage: "stable_audio" },
            { chapterId: "chapter_003", title: "第三章", position: 2, status: "queued" },
          ],
        }}
      />,
    );

    expect(screen.getByText("1 / 3 章已处理")).toBeInTheDocument();
    expect(screen.getByText("第一章")).toBeInTheDocument();
    expect(screen.getByText("第二章")).toBeInTheDocument();
    expect(screen.getByText("正在Stable Audio 背景音/音效")).toBeInTheDocument();
    expect(screen.getByText("排队中")).toBeInTheDocument();
  });

  it("distinguishes the serialized MiMo lane from later durable stages", () => {
    render(
      <BatchGenerationStatusList
        batch={{
          batchId: "batch_1",
          status: "running",
          mimoCooldownSeconds: 5,
          chapters: [
            {
              chapterId: "chapter_001",
              title: "第一章",
              position: 0,
              status: "queued",
              nextStage: "voice_synthesize",
              stageState: "ready",
            },
            {
              chapterId: "chapter_002",
              title: "第二章",
              position: 1,
              status: "running",
              nextStage: "voice_synthesize",
              stageState: "running",
            },
            {
              chapterId: "chapter_003",
              title: "第三章",
              position: 2,
              status: "running",
              nextStage: "audio_plan",
              stageState: "ready",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("等待 MiMo")).toBeInTheDocument();
    expect(screen.getByText("MiMo 配音中")).toBeInTheDocument();
    expect(screen.getByText("MiMo 限流冷却中，约5 秒后继续")).toBeInTheDocument();
    expect(screen.getByText("正在原章节配音（MiMo 串行）")).toBeInTheDocument();
    expect(screen.getByText("等待背景音/音效规划")).toBeInTheDocument();
  });

  it("shows total elapsed time from submission and refreshes it while the batch is active", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-12T02:00:03.000Z"));
    render(
      <BatchGenerationStatusList
        batch={{
          batchId: "batch_1",
          status: "queued",
          createdAt: 1_786_500_000,
          chapters: [
            { chapterId: "chapter_001", title: "第一章", position: 0, status: "queued" },
          ],
        }}
      />,
    );

    expect(screen.getByText("总耗时：3 秒")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(2_000));
    expect(screen.getByText("总耗时：5 秒")).toBeInTheDocument();
  });

  it("keeps the final elapsed time from the backend completion timestamp", () => {
    expect(formatBatchElapsed(0)).toBe("0 秒");
    expect(formatBatchElapsed(65)).toBe("1 分 05 秒");
    expect(formatBatchElapsed(3_661)).toBe("1 小时 01 分 01 秒");
    expect(batchElapsedSeconds({ createdAt: 100, completedAt: 165 }, 999_999)).toBe(65);

    render(
      <BatchGenerationStatusList
        batch={{
          status: "succeeded",
          createdAt: 100,
          completedAt: 165,
          chapters: [
            { chapterId: "chapter_001", title: "第一章", position: 0, status: "succeeded" },
          ],
        }}
      />,
    );

    expect(screen.getByText("总耗时：1 分 05 秒")).toBeInTheDocument();
  });

  it("shows the per-chapter failure reason", () => {
    render(
      <BatchGenerationStatusList
        batch={{
          status: "completed_with_errors",
          chapters: [
            {
              chapterId: "chapter_001",
              title: "第一章",
              position: 0,
              status: "failed",
              error: "LLM 请求超时",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("已完成（有失败）")).toBeInTheDocument();
    expect(screen.getByText("LLM 请求超时")).toBeInTheDocument();
  });

  it("shows persisted chapter duration and exposes per-stage timing details", () => {
    render(
      <BatchGenerationStatusList
        batch={{
          status: "succeeded",
          chapters: [{
            chapterId: "chapter_001",
            title: "第一章",
            position: 0,
            status: "succeeded",
            durationSeconds: 75,
            stageTimings: { voice: 25, transcript: 10, audio_plan: 40 },
          }],
        }}
      />,
    );

    const duration = screen.getByText("耗时 1 分 15 秒");
    expect(duration).toBeInTheDocument();
    expect(duration).toHaveAttribute("title", expect.stringContaining("原章节配音：25 秒"));
    expect(duration).toHaveAttribute("title", expect.stringContaining("背景音/音效规划：40 秒"));
  });
});
