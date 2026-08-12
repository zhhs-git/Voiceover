"""Durable, human-readable workflow progress for the desktop and LAN UI.

Worker calls are intentionally still synchronous.  This module only records
the current stage in the chapter's existing ``analysis/state.json`` file so a
second request can inspect progress while the original worker is running.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


ANALYSIS_WORKFLOW_STEPS = (
    "characters",
    "voice_design",
    "speakers",
    "delivery",
    "voice_direction",
    "script",
)
GENERATION_WORKFLOW_STEPS = ("voice", "transcript", "audio_plan", "stable_audio", "mix")


def _new_workflow(step_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "status": "pending",
        "currentStep": None,
        "steps": {
            step_id: {"status": "pending"} for step_id in step_ids
        },
        "updatedAt": time.time(),
    }


def start_workflow(
    state: dict[str, Any],
    kind: str,
    step_ids: tuple[str, ...],
    *,
    first_step: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    workflow = _new_workflow(step_ids)
    workflow["status"] = "running" if first_step else "pending"
    workflow["currentStep"] = first_step
    if detail:
        workflow["detail"] = detail
    if first_step:
        workflow["steps"][first_step] = {"status": "running"}
    state.setdefault("workflow", {})[kind] = workflow
    return state


def update_workflow(
    state: dict[str, Any],
    kind: str,
    step_ids: tuple[str, ...],
    *,
    step: str | None = None,
    step_status: str | None = None,
    status: str | None = None,
    detail: str | None = None,
    error: str | None = None,
    current_step: str | None = None,
) -> dict[str, Any]:
    workflows = state.setdefault("workflow", {})
    workflow = workflows.get(kind)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("steps"), dict):
        start_workflow(state, kind, step_ids)
        workflow = state["workflow"][kind]

    steps = workflow["steps"]
    for step_id in step_ids:
        if not isinstance(steps.get(step_id), dict):
            steps[step_id] = {"status": "pending"}

    if step and step not in step_ids:
        raise ValueError(f"unknown {kind} workflow step: {step}")
    if step and step_status:
        entry = steps[step]
        entry["status"] = step_status
        if detail:
            entry["detail"] = detail
        elif step_status != "failed":
            entry.pop("detail", None)
        if error:
            entry["error"] = error
        else:
            entry.pop("error", None)

    if current_step is not None:
        if current_step not in step_ids:
            raise ValueError(f"unknown {kind} workflow current step: {current_step}")
        workflow["currentStep"] = current_step
    elif step and step_status in {"running", "needs_review", "failed"}:
        workflow["currentStep"] = step

    if detail:
        workflow["detail"] = detail
    if error:
        workflow["error"] = error
    elif status != "failed":
        workflow.pop("error", None)

    if status:
        workflow["status"] = status
    elif step_status == "failed":
        workflow["status"] = "failed"
    elif step_status == "needs_review":
        workflow["status"] = "needs_review"
    elif step_status == "running":
        workflow["status"] = "running"
    elif all(
        isinstance(steps[step_id], dict)
        and steps[step_id].get("status") in {"succeeded", "skipped"}
        for step_id in step_ids
    ):
        workflow["status"] = "succeeded"
        workflow["currentStep"] = None
    elif step_status == "succeeded" and workflow.get("status") not in {
        "failed",
        "needs_review",
    }:
        workflow["status"] = "running"

    workflow["updatedAt"] = time.time()
    return state


def read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def update_state_file(
    path: Path,
    kind: str,
    step_ids: tuple[str, ...],
    **kwargs: Any,
) -> dict[str, Any]:
    state = read_state(path)
    update_workflow(state, kind, step_ids, **kwargs)
    write_state(path, state)
    return state
