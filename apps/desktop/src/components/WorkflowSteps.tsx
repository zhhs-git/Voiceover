import type { ChapterWorkflowStatus, WorkflowStepStatus } from "../types";

const STATUS_LABELS: Record<WorkflowStepStatus, string> = {
  pending: "待处理",
  running: "进行中",
  succeeded: "已完成",
  failed: "失败",
  needs_review: "等待操作",
  skipped: "已跳过",
};

const STATUS_SYMBOLS: Record<WorkflowStepStatus, string> = {
  pending: "○",
  running: "◌",
  succeeded: "✓",
  failed: "×",
  needs_review: "!",
  skipped: "–",
};

interface WorkflowStepsProps {
  workflow: ChapterWorkflowStatus | undefined;
  title: string;
  compact?: boolean;
}

export function WorkflowSteps({ workflow, title, compact = false }: WorkflowStepsProps) {
  if (!workflow) return null;
  return (
    <section className={`workflow-card ${compact ? "workflow-card-compact" : ""}`}>
      <div className="workflow-card-header">
        <div>
          <p className="workflow-card-title">{title}</p>
          {workflow.detail && <p className="workflow-card-detail">{workflow.detail}</p>}
        </div>
        <span className={`workflow-overall workflow-${workflow.status}`}>
          {STATUS_LABELS[workflow.status]}
        </span>
      </div>
      <div className="workflow-steps" role="list" aria-label={`${title}阶段`}>
        {workflow.steps.map((step, index) => (
          <div
            className={`workflow-step workflow-${step.status}${
              workflow.currentStep === step.id ? " workflow-step-current" : ""
            }`}
            key={step.id}
            role="listitem"
          >
            <span className="workflow-step-marker" aria-hidden="true">
              {STATUS_SYMBOLS[step.status]}
            </span>
            <span className="workflow-step-content">
              <span className="workflow-step-name">
                {index + 1}. {step.label}
              </span>
              <span className="workflow-step-status">{STATUS_LABELS[step.status]}</span>
              {step.detail && <span className="workflow-step-detail">{step.detail}</span>}
              {step.error && <span className="workflow-step-error">{step.error}</span>}
            </span>
          </div>
        ))}
      </div>
      {workflow.error && !workflow.steps.some((step) => step.error === workflow.error) && (
        <p className="workflow-error">{workflow.error}</p>
      )}
    </section>
  );
}

export function workflowStatusLabel(status: WorkflowStepStatus): string {
  return STATUS_LABELS[status];
}

