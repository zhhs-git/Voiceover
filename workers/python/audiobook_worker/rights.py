from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RightsClassification = Literal["allowed", "restricted", "unknown", "blocked"]


@dataclass(frozen=True)
class RightsClassificationResult:
    classification: RightsClassification
    reason: str
    requires_attestation: bool
    evidence: list[str] = field(default_factory=list)


def classify_rights(
    input_path: Path | str,
    *,
    metadata: dict[str, Any],
) -> RightsClassificationResult:
    if metadata.get("drm") is True:
        return RightsClassificationResult(
            classification="blocked",
            reason="drm_detected",
            requires_attestation=False,
            evidence=["metadata.drm=true"],
        )

    text = _read_sample(Path(input_path)).lower()
    rights_text = " ".join(str(value).lower() for value in metadata.values())
    combined = f"{rights_text}\n{text}"

    if "project gutenberg" in combined or "public domain" in combined:
        return RightsClassificationResult(
            classification="allowed",
            reason="public_domain_notice",
            requires_attestation=False,
            evidence=["public_domain_notice"],
        )

    if "creative commons" in combined or "cc by" in combined or "cc0" in combined:
        return RightsClassificationResult(
            classification="allowed",
            reason="creative_commons",
            requires_attestation=False,
            evidence=["creative_commons_notice"],
        )

    if "all rights reserved" in combined:
        return RightsClassificationResult(
            classification="restricted",
            reason="all_rights_reserved_notice",
            requires_attestation=True,
            evidence=["all_rights_reserved"],
        )

    return RightsClassificationResult(
        classification="unknown",
        reason="missing_rights_metadata",
        requires_attestation=True,
        evidence=[],
    )


def _read_sample(path: Path) -> str:
    suffix = path.suffix.lower()
    # Binary formats: extract the first 20KB of decodable text; skip if
    # the file is pure binary (EPUB, PDF) to avoid garbage in the combined
    # rights text that could produce false negatives.
    if suffix in (".epub", ".pdf", ".mobi", ".azw", ".azw3"):
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:20_000]
    except OSError:
        return ""
