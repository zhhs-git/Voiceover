import type { ProgressDetail } from "../types";

export function generationProgressDetails({
  now,
  startTime,
  doneSegments,
  totalSegments,
  chapterIndex,
  chapterCount,
  segmentCount,
}: {
  now: number;
  startTime: number;
  doneSegments: number;
  totalSegments: number;
  chapterIndex: number;
  chapterCount: number;
  segmentCount: number;
}): ProgressDetail[] {
  const elapsed = Math.round((now - startTime) / 1000);
  const avgPerSeg = doneSegments > 0 ? elapsed / doneSegments : 8;
  const remaining = Math.round(avgPerSeg * Math.max(totalSegments - doneSegments, 0));

  return [
    { label: "后端", value: "MiMo V2.5 TTS 音色设计" },
    { label: "章节", value: `第 ${chapterIndex} / ${chapterCount} 章` },
    { label: "片段", value: String(segmentCount) },
    { label: "已用时", value: `${elapsed} 秒` },
    { label: "预计剩余", value: `约 ${remaining} 秒` },
  ];
}
