"""Stable, chapter-scoped identifiers for generated audio assets."""

from __future__ import annotations

import re
from typing import Any


def _id_prefix(value: object, fallback: str) -> str:
    """Make a scene name safe to use as an asset-id namespace."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return normalized or fallback


def _allocate_unique_id(
    raw_id: object,
    *,
    scene_id: object,
    fallback: str,
    used_ids: set[str],
) -> str:
    """Keep the first valid ID and namespace later collisions by scene."""

    base = str(raw_id or "").strip()
    if not base:
        return base
    if base not in used_ids:
        used_ids.add(base)
        return base

    prefix = _id_prefix(scene_id, fallback)
    candidate = f"{prefix}_{base}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"{prefix}_{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def normalize_audio_plan_asset_ids(audio_plan: dict[str, Any]) -> list[str]:
    """Make music variants and SFX IDs unique across one chapter.

    LLMs commonly restart numbering inside every scene (for example, each
    scene emits ``sfx_1``).  Asset manifests are chapter-wide, so those IDs
    must be unique.  The first occurrence keeps its legacy ID for cache
    compatibility; later collisions receive a deterministic scene namespace.
    The plan is normalized in place and the returned messages describe any
    changes made.
    """

    raw_scenes = audio_plan.get("scenes", [])
    if not isinstance(raw_scenes, list):
        return []

    used_music_ids: set[str] = set()
    used_sfx_ids: set[str] = set()
    changes: list[str] = []

    for scene_index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            continue
        scene_id = raw_scene.get("id") or f"scene_{scene_index + 1:03d}"
        scene_label = str(scene_id)

        raw_variants = raw_scene.get("musicVariants", [])
        variant_mapping: dict[str, str] = {}
        if isinstance(raw_variants, list):
            for raw_variant in raw_variants:
                if not isinstance(raw_variant, dict):
                    continue
                old_id = str(raw_variant.get("id") or "").strip()
                new_id = _allocate_unique_id(
                    old_id,
                    scene_id=scene_id,
                    fallback=f"scene_{scene_index + 1:03d}",
                    used_ids=used_music_ids,
                )
                if old_id and old_id not in variant_mapping:
                    variant_mapping[old_id] = new_id
                if new_id and new_id != old_id:
                    raw_variant["id"] = new_id
                    changes.append(
                        f"music:{old_id}->{new_id}@{scene_label}"
                    )

        # Cues are the only references to music variant IDs.  Keep them in
        # sync when a duplicate variant was namespaced.
        raw_cues = raw_scene.get("musicCues", [])
        if isinstance(raw_cues, list):
            for raw_cue in raw_cues:
                if not isinstance(raw_cue, dict):
                    continue
                old_id = str(raw_cue.get("variantId") or "").strip()
                new_id = variant_mapping.get(old_id)
                if new_id and new_id != old_id:
                    raw_cue["variantId"] = new_id

        raw_sfx = raw_scene.get("sfx", [])
        if isinstance(raw_sfx, list):
            for raw_effect in raw_sfx:
                if not isinstance(raw_effect, dict):
                    continue
                old_id = str(raw_effect.get("id") or "").strip()
                new_id = _allocate_unique_id(
                    old_id,
                    scene_id=scene_id,
                    fallback=f"scene_{scene_index + 1:03d}",
                    used_ids=used_sfx_ids,
                )
                if new_id and new_id != old_id:
                    raw_effect["id"] = new_id
                    changes.append(f"sfx:{old_id}->{new_id}@{scene_label}")

    return changes


def normalize_script_audio_asset_ids(script: dict[str, Any]) -> list[str]:
    """Normalize the audio plan embedded in a chapter script."""

    raw_plan = script.get("audioPlan")
    if not isinstance(raw_plan, dict):
        return []
    return normalize_audio_plan_asset_ids(raw_plan)
