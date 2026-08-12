import { invoke } from "./platform";
import type {
  ChapterWorkflowStatus,
  WorkflowStep,
  WorkflowStepStatus,
} from "../types";
import type {
  BatchGenerationChapter,
  BatchGenerationResponse,
} from "./batchGeneration";

export type WorkflowKind = "analysis" | "generation";

interface WorkflowDefinition {
  id: string;
  label: string;
}

export const ANALYSIS_WORKFLOW_DEFINITION: WorkflowDefinition[] = [
  { id: "characters", label: "角色身份分析" },
  { id: "voice_design", label: "角色音色设计" },
  { id: "speakers", label: "说话人归属" },
  { id: "delivery", label: "情绪与语速分析" },
  { id: "voice_direction", label: "动态演绎指导" },
  { id: "script", label: "脚本组装" },
];

export const GENERATION_WORKFLOW_DEFINITION: WorkflowDefinition[] = [
  { id: "voice", label: "原章节配音" },
  { id: "transcript", label: "Whisper 转录" },
  { id: "audio_plan", label: "音频规划" },
  { id: "stable_audio", label: "Stable Audio 背景音/音效" },
  { id: "mix", label: "最终混音" },
];

/**
 * The durable batch queue is the first source to know a chapter has reached
 * a terminal state. Convert that response into the same shape used by the
 * workflow UI so completion is visible before the next state.json poll.
 */
export function generationWorkflowFromBatchChapter(
  chapter: BatchGenerationChapter,
  updatedAt?: number,
): ChapterWorkflowStatus | null {
  if (chapter.status !== "succeeded" && chapter.status !== "failed") return null;

  const failureIndex = chapter.status === "failed"
    ? GENERATION_WORKFLOW_DEFINITION.findIndex((step) => step.id === chapter.currentStage)
    : -1;
  const error = chapter.error || "批量生成失败。";

  return {
    chapterId: chapter.chapterId,
    kind: "generation",
    currentStep:
      chapter.status === "failed" && failureIndex >= 0
        ? GENERATION_WORKFLOW_DEFINITION[failureIndex].id
        : undefined,
    steps: GENERATION_WORKFLOW_DEFINITION.map((step, index) => {
      if (chapter.status === "succeeded") {
        return { ...step, status: "succeeded" as const };
      }
      if (index < failureIndex) {
        return { ...step, status: "succeeded" as const };
      }
      if (index === failureIndex) {
        return { ...step, status: "failed" as const, error };
      }
      return { ...step, status: "pending" as const };
    }),
    status: chapter.status === "succeeded" ? "succeeded" : "failed",
    detail: chapter.status === "succeeded" ? "批量生成已完成。" : undefined,
    error: chapter.status === "failed" ? error : undefined,
    updatedAt,
  };
}

export function terminalGenerationWorkflowsFromBatch(
  batch: Pick<BatchGenerationResponse, "chapters" | "updatedAt">,
): Record<string, ChapterWorkflowStatus> {
  const workflows: Record<string, ChapterWorkflowStatus> = {};
  for (const chapter of batch.chapters) {
    const workflow = generationWorkflowFromBatchChapter(chapter, batch.updatedAt);
    if (workflow) workflows[chapter.chapterId] = workflow;
  }
  return workflows;
}

const VALID_STATUSES = new Set<WorkflowStepStatus>([
  "pending",
  "running",
  "succeeded",
  "failed",
  "needs_review",
  "skipped",
]);

function statusOf(value: unknown): WorkflowStepStatus {
  return typeof value === "string" && VALID_STATUSES.has(value as WorkflowStepStatus)
    ? (value as WorkflowStepStatus)
    : "pending";
}

function definitionFor(kind: WorkflowKind): WorkflowDefinition[] {
  return kind === "analysis"
    ? ANALYSIS_WORKFLOW_DEFINITION
    : GENERATION_WORKFLOW_DEFINITION;
}

function fallbackStatus(
  raw: Record<string, unknown>,
  kind: WorkflowKind,
  stepId: string,
): WorkflowStepStatus {
  if (
    kind === "analysis" &&
    (stepId === "voice_design" || stepId === "voice_direction") &&
    raw.script &&
    typeof raw.script === "object" &&
    (raw.script as Record<string, unknown>).status === "succeeded"
  ) {
    // Older state files predate these internal stages. A completed legacy
    // script is the only durable evidence available, so treat the new
    // stages as completed for display compatibility.
    return "succeeded";
  }
  const sourceKey = kind === "generation"
    ? ({ audio_plan: "audioPlan", stable_audio: "stableAudio", mix: "mixed" } as Record<string, string>)[stepId] ?? stepId
    : stepId;
  const source = raw[sourceKey];
  if (source && typeof source === "object") {
    const status = statusOf((source as Record<string, unknown>).status);
    if (status !== "pending") return status;
  }
  if (stepId === "script" && raw.script && typeof raw.script === "object") {
    return statusOf((raw.script as Record<string, unknown>).status);
  }
  return "pending";
}

function overallStatus(steps: WorkflowStep[]): WorkflowStepStatus {
  if (steps.some((step) => step.status === "failed")) return "failed";
  if (steps.some((step) => step.status === "needs_review")) return "needs_review";
  if (steps.some((step) => step.status === "running")) return "running";
  if (steps.every((step) => step.status === "succeeded" || step.status === "skipped")) {
    return "succeeded";
  }
  return "pending";
}

export function parseChapterWorkflow(
  rawValue: unknown,
  chapterId: string,
  kind: WorkflowKind,
): ChapterWorkflowStatus {
  const raw = rawValue && typeof rawValue === "object"
    ? rawValue as Record<string, unknown>
    : {};
  const workflowRoot = raw.workflow && typeof raw.workflow === "object"
    ? raw.workflow as Record<string, unknown>
    : {};
  const workflow = workflowRoot[kind] && typeof workflowRoot[kind] === "object"
    ? workflowRoot[kind] as Record<string, unknown>
    : {};
  const storedSteps = workflow.steps && typeof workflow.steps === "object"
    ? workflow.steps as Record<string, unknown>
    : {};
  const definitions = definitionFor(kind);
  const steps: WorkflowStep[] = definitions.map(({ id, label }) => {
    const stored = storedSteps[id] && typeof storedSteps[id] === "object"
      ? storedSteps[id] as Record<string, unknown>
      : {};
    const status = Object.keys(stored).length > 0
      ? statusOf(stored.status)
      : fallbackStatus(raw, kind, id);
    return {
      id,
      label,
      status,
      detail: typeof stored.detail === "string" ? stored.detail : undefined,
      error: typeof stored.error === "string" ? stored.error : undefined,
    };
  });
  const storedStatus = statusOf(workflow.status);
  const status = storedStatus !== "pending" || Object.keys(workflow).length > 0
    ? (storedStatus === "pending" ? overallStatus(steps) : storedStatus)
    : overallStatus(steps);
  const currentStep = typeof workflow.currentStep === "string"
    ? workflow.currentStep
    : steps.find((step) => step.status === "running" || step.status === "needs_review")?.id;
  const failedStep = steps.find((step) => step.status === "failed");
  return {
    chapterId,
    kind,
    currentStep,
    steps,
    status,
    detail: typeof workflow.detail === "string" ? workflow.detail : undefined,
    error: typeof workflow.error === "string" ? workflow.error : failedStep?.error,
    updatedAt: typeof workflow.updatedAt === "number" ? workflow.updatedAt : undefined,
  };
}

export function emptyChapterWorkflow(
  chapterId: string,
  kind: WorkflowKind,
): ChapterWorkflowStatus {
  return parseChapterWorkflow({}, chapterId, kind);
}

export async function readChapterWorkflow(
  workDir: string,
  chapterId: string,
  kind: WorkflowKind,
): Promise<ChapterWorkflowStatus> {
  const statePath = `${workDir}/analysis/${chapterId}/state.json`;
  try {
    const raw = await invoke<string>("run_worker", {
      command: "_read_file",
      inputJson: JSON.stringify({ path: statePath }),
    });
    return parseChapterWorkflow(JSON.parse(raw), chapterId, kind);
  } catch {
    return emptyChapterWorkflow(chapterId, kind);
  }
}

export function watchChapterWorkflow(
  workDir: string,
  chapterId: string,
  kind: WorkflowKind,
  onUpdate: (status: ChapterWorkflowStatus) => void,
  intervalMs = 750,
): () => void {
  let stopped = false;
  const poll = async () => {
    const status = await readChapterWorkflow(workDir, chapterId, kind);
    if (!stopped) onUpdate(status);
  };
  void poll();
  const timer = window.setInterval(() => void poll(), intervalMs);
  return () => {
    stopped = true;
    window.clearInterval(timer);
  };
}
