from pathlib import Path

from audiobook_worker.workflow import (
    ANALYSIS_WORKFLOW_STEPS,
    GENERATION_WORKFLOW_STEPS,
    start_workflow,
    update_state_file,
    update_workflow,
)


def test_analysis_workflow_advances_and_completes() -> None:
    state: dict[str, object] = {}
    start_workflow(state, "analysis", ANALYSIS_WORKFLOW_STEPS, first_step="characters")
    update_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        step="characters",
        step_status="succeeded",
    )
    update_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        step="voice_design",
        step_status="succeeded",
    )
    update_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        step="speakers",
        step_status="running",
    )

    analysis = state["workflow"]["analysis"]  # type: ignore[index]
    assert analysis["status"] == "running"
    assert analysis["currentStep"] == "speakers"
    assert analysis["steps"]["characters"]["status"] == "succeeded"

    for step in ("speakers", "delivery", "voice_direction", "script"):
        update_workflow(
            state,
            "analysis",
            ANALYSIS_WORKFLOW_STEPS,
            step=step,
            step_status="succeeded",
        )
    assert state["workflow"]["analysis"]["status"] == "succeeded"  # type: ignore[index]


def test_workflow_state_file_is_readable_after_atomic_update(tmp_path: Path) -> None:
    state_path = tmp_path / "analysis" / "chapter_1" / "state.json"
    update_state_file(
        state_path,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        step="stable_audio",
        step_status="needs_review",
        detail="等待试听并点击下一个",
    )

    import json

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["workflow"]["generation"]["status"] == "needs_review"
    assert payload["workflow"]["generation"]["steps"]["stable_audio"]["detail"] == "等待试听并点击下一个"
