import { describe, expect, it } from "vitest";

import { generationProgressDetails } from "./generationProgress";

describe("generationProgressDetails", () => {
  it("computes elapsed from the current clock value", () => {
    const details = generationProgressDetails({
      now: 7_500,
      startTime: 1_000,
      doneSegments: 2,
      totalSegments: 6,
      chapterIndex: 1,
      chapterCount: 3,
      segmentCount: 10,
    });

    expect(details).toContainEqual({ label: "已用时", value: "7 秒" });
    expect(details).toContainEqual({ label: "预计剩余", value: "约 14 秒" });
  });

  it("does not hardcode a TTS backend in progress details", () => {
    const details = generationProgressDetails({
      now: 5_000,
      startTime: 1_000,
      doneSegments: 1,
      totalSegments: 3,
      chapterIndex: 1,
      chapterCount: 1,
      segmentCount: 3,
    });

    expect(details.find((detail) => detail.label === "配音模型")?.value).toBe(
      "按当前模型配置",
    );
  });
});
