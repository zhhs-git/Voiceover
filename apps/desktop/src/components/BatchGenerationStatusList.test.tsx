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
});
