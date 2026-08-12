import { describe, expect, it } from "vitest";
import { parseChapterWorkflow } from "./workflowStatus";

describe("parseChapterWorkflow", () => {
  it("parses durable analysis stages and exposes the current step", () => {
    const status = parseChapterWorkflow(
      {
        workflow: {
          analysis: {
            status: "running",
            currentStep: "speakers",
            steps: {
              characters: { status: "succeeded" },
              speakers: { status: "running", detail: "正在判断对白归属" },
              delivery: { status: "pending" },
              script: { status: "pending" },
            },
          },
        },
      },
      "chapter_1",
      "analysis",
    );

    expect(status.status).toBe("running");
    expect(status.currentStep).toBe("speakers");
    expect(status.steps.map((step) => step.status)).toEqual([
      "succeeded",
      "pending",
      "running",
      "pending",
      "pending",
      "pending",
    ]);
    expect(status.steps[2].label).toBe("说话人归属");
  });

  it("preserves the Stable Audio needs-review state", () => {
    const status = parseChapterWorkflow(
      {
        workflow: {
          generation: {
            status: "needs_review",
            currentStep: "stable_audio",
            steps: {
              voice: { status: "succeeded" },
              transcript: { status: "succeeded" },
              audio_plan: { status: "succeeded" },
              stable_audio: {
                status: "needs_review",
                detail: "等待试听并点击下一个",
              },
              mix: { status: "pending" },
            },
          },
        },
      },
      "chapter_1",
      "generation",
    );

    expect(status.status).toBe("needs_review");
    expect(status.currentStep).toBe("stable_audio");
    expect(status.steps[3].detail).toContain("试听");
  });

  it("falls back to legacy persisted stage fields", () => {
    const status = parseChapterWorkflow(
      {
        characters: { status: "succeeded" },
        speakers: { status: "succeeded" },
        delivery: { status: "succeeded" },
        script: { status: "succeeded" },
      },
      "chapter_1",
      "analysis",
    );

    expect(status.status).toBe("succeeded");
    expect(status.steps.every((step) => step.status === "succeeded")).toBe(true);
  });
});
