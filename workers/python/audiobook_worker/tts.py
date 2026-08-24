from __future__ import annotations

import base64
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from audiobook_worker.dialogue import resolve_text_language
from audiobook_worker.model_settings import (
    VOXCPM2_MODEL_ID,
    voxcpm2_paths,
)
from audiobook_worker.script_builder import (
    DEFAULT_NARRATOR_VOICE_ID,
    VOICE_REGISTRY,
    is_narrator_voice_id,
    normalize_narrator_voice_id,
)
from audiobook_worker.voxcpm2_profile_loudness import (
    profile_loudness_is_current,
    voxcpm2_profile_loudness,
)
from audiobook_worker.tts_quality import (
    TtsSegmentAudioQualityError,
    TtsSegmentAudioQualityResult,
    maximum_duration_for_segment,
    validate_tts_segment_wav,
)

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

_EMOTION_MODIFIERS: dict[str, str] = {
    "neutral": "The speaker delivers the text clearly and evenly, without notable emotional colour.",
    "happy": "The speaker sounds genuinely warm and cheerful, voice naturally lifted; do not overplay it.",
    "sad": "The speaker sounds sorrowful and subdued, voice slightly heavy, sentence-ends trailing off.",
    "angry": "The speaker sounds angry and forceful, sharp emphasis and barely contained intensity; do not underplay the emotion.",
    "afraid": "The speaker sounds frightened and unsteady, voice slightly trembling, breath catching on key words.",
    "tense": "The speaker sounds tense and guarded, every word deliberate and measured, as if braced for danger.",
    "teasing": "The speaker sounds playfully mocking, a sly edge to every word; do not let it slide into sincerely cheerful.",
    "whispering": "The speaker whispers with audible breath and hushed intimacy, clearly reduced in volume, as though sharing a secret.",
    "excited": "The speaker sounds enthusiastic and energised, slightly faster, barely able to contain the feeling.",
    "tired": "The speaker sounds weary and drained, delivery heavy and low-energy, each word an effort.",
    "grief": "The speaker sounds deeply grief-stricken, voice cracking and thick with tears, far beyond ordinary sadness.",
    "cold": "The speaker sounds emotionally cold and detached, flat affect, deliberately distant, no warmth whatsoever.",
    "pleading": "The speaker pleads with desperate urgency, voice softened and slightly trembling, vulnerability fully exposed.",
    "surprised": "The speaker sounds genuinely startled, voice pitching upward in sudden disbelief — shocked, not joyful.",
    "gentle": "The speaker sounds tender and soothing, slow and soft, as if comforting someone fragile or very young.",
    "resolute": "The speaker sounds firm and unwavering, each word carrying deliberate weight and absolute conviction.",
    "nervous": "The speaker sounds anxious and unsettled, slight vocal trembling, pauses slightly too long, clearly ill at ease.",
    "contemptuous": "The speaker sounds coldly contemptuous and dismissive, no humour, looking down from a position of superiority.",
    "solemn": "The speaker sounds grave and ceremonious, measured and unhurried, befitting an oath or a momentous declaration.",
    "bitter": "The speaker sounds quietly bitter and resentful, suppressed indignation beneath each word, grievances swallowed but unmistakable.",
}

_PACE_MODIFIERS: dict[str, str] = {
    "slow": "The pace is slow and unhurried.",
    "normal": "",
    "fast": "The pace is quick and urgent.",
}

_DEFAULT_MODEL_ID = "parler-tts/parler-tts-mini-v1"
_MIMO_ENDPOINT = "https://api.xiaomimimo.com/v1/chat/completions"
_MIMO_VOICE_DESIGN_MODEL_ID = "mimo-v2.5-tts-voicedesign"
_MIMO_VOICE_CLONE_MODEL_ID = "mimo-v2.5-tts-voiceclone"
# Voice cloning is the safe default: a caller must explicitly opt into
# voice-design when it is generating a reference profile.
_MIMO_MODEL_ID = _MIMO_VOICE_CLONE_MODEL_ID
_MIMO_KEYCHAIN_SERVICE = "audiobook-generator.mimo-api-key"
_MIMO_VOICE_PROFILE_VERSION = 1
_MIMO_REFERENCE_TEXT = (
    "这是一段稳定的声音样本。请保持自然、清晰、连贯地说完这段话，"
    "不要刻意表演，也不要改变自己的基础音色。"
)
_MIMO_MAX_REFERENCE_BASE64_LENGTH = 10 * 1024 * 1024
_MIMO_SAFE_RPM = 80
_MIMO_RATE_STATE_ENV = "AUDIOBOOK_MIMO_RATE_STATE_PATH"
VOXCPM2_PROMPT_FORMAT_VERSION = 2
_VOXCPM2_VOICE_PROFILE_VERSION = VOXCPM2_PROMPT_FORMAT_VERSION
_VOXCPM2_REFERENCE_TEXTS = {
    "zh": "清晨的风穿过窗边，屋里很安静。",
    "en": "The morning light falls softly across the quiet room.",
}
_VOXCPM2_PROFILE_CONTROL_MAX_CHARACTERS = 180
_VOXCPM2_DIRECTION_MAX_CHARACTERS = 120
_VOXCPM2_RUNNER_TIMEOUT_SECONDS = 60 * 60


class MiMoRequestError(RuntimeError):
    """A MiMo request failure with an explicit retry policy for callers."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        quality: TtsSegmentAudioQualityResult | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.quality = quality


def _bounded_positive_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Parse a runtime integer setting without allowing unsafe values."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _mimo_max_attempts() -> int:
    return _bounded_positive_int(
        os.environ.get("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "3"),
        default=3,
        minimum=1,
        maximum=5,
    )


def mimo_tts_concurrency() -> int:
    """MiMo cloud synthesis is intentionally serial for the whole service."""
    return 1


def mimo_tts_rpm() -> int:
    """Return the conservative global request-start budget for MiMo TTS."""
    return _bounded_positive_int(
        os.environ.get("AUDIOBOOK_MIMO_RPM", str(_MIMO_SAFE_RPM)),
        default=_MIMO_SAFE_RPM,
        minimum=1,
        maximum=_MIMO_SAFE_RPM,
    )


def _mimo_retry_backoff_seconds() -> float:
    try:
        value = float(os.environ.get("AUDIOBOOK_MIMO_RETRY_BACKOFF_SECONDS", "0.75"))
    except (TypeError, ValueError):
        value = 0.75
    return max(0.0, min(10.0, value))


def _mimo_rate_state_path() -> Path:
    """Locate the state file shared by all MiMo worker child processes."""
    configured = os.environ.get(_MIMO_RATE_STATE_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "audiobook-generator-mimo-rate.json"


def _read_mimo_rate_state(state_file) -> dict[str, float]:
    state_file.seek(0)
    try:
        raw = json.loads(state_file.read() or "{}")
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        return {}
    state: dict[str, float] = {}
    for key in ("lastStartMonotonic", "cooldownUntilMonotonic"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            state[key] = float(value)
    return state


def _write_mimo_rate_state(state_file, state: dict[str, float]) -> None:
    state_file.seek(0)
    state_file.truncate()
    json.dump(state, state_file, separators=(",", ":"), sort_keys=True)
    state_file.flush()
    os.fsync(state_file.fileno())


class _MiMoRequestRateGate:
    """Cross-process serial gate and no-burst rate governor for real requests."""

    def __init__(self, state_file) -> None:
        self._state_file = state_file
        self._state = _read_mimo_rate_state(state_file)

    def wait_for_turn(self) -> None:
        now = time.monotonic()
        last_start = self._state.get("lastStartMonotonic", 0.0)
        cooldown_until = self._state.get("cooldownUntilMonotonic", 0.0)
        # A persisted monotonic timestamp from a previous host boot is not
        # comparable with this process. Treat an implausibly distant value as
        # stale rather than blocking the service indefinitely.
        if last_start > now + 300:
            last_start = 0.0
        if cooldown_until > now + 300:
            cooldown_until = 0.0
        interval = 60.0 / mimo_tts_rpm()
        start_at = max(last_start + interval, cooldown_until)
        if start_at > now:
            time.sleep(start_at - now)
        self._state["lastStartMonotonic"] = time.monotonic()
        _write_mimo_rate_state(self._state_file, self._state)

    def set_cooldown(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            return
        deadline = time.monotonic() + seconds
        self._state["cooldownUntilMonotonic"] = max(
            self._state.get("cooldownUntilMonotonic", 0.0),
            deadline,
        )
        _write_mimo_rate_state(self._state_file, self._state)


@contextmanager
def _mimo_request_rate_gate():
    """Hold the only MiMo HTTP lane for one logical request/retry sequence."""
    state_path = _mimo_rate_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a+", encoding="utf-8") as state_file:
        fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
        try:
            yield _MiMoRequestRateGate(state_file)
        finally:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)


def _retry_after_seconds(value: object) -> float | None:
    """Read a standard Retry-After seconds value or HTTP-date."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, when.timestamp() - time.time())


def _mimo_retry_delay_seconds(
    attempt: int,
    *,
    retry_after: float | None = None,
    rate_limited: bool = False,
) -> float:
    """Produce bounded, jittered retry delay and honor provider cooldowns."""
    exponential = _mimo_retry_backoff_seconds() * (2 ** max(0, attempt - 1))
    if rate_limited:
        exponential = max(5.0, exponential)
    base = max(exponential, retry_after or 0.0)
    if base <= 0:
        return 0.0
    return min(300.0, base + random.uniform(0.0, min(1.0, base * 0.1)))

_MIMO_VOICE_DESIGNS: dict[str, str] = {
    "narrator_default": "角色：一位专业中文有声书旁白，固定为同一位成年女性。声音洪亮饱满而温暖、柔和、清晰，气息稳定，胸腔共鸣自然，口腔共鸣自然，咬字清楚；保持稳定统一的旁白声线，连贯耐听，不代入任何角色，不使用夸张播音腔。",
    "narrator_female": "角色：一位专业中文有声书旁白，固定为同一位成年女性。声音洪亮饱满而温暖、柔和、清晰，气息稳定，胸腔共鸣自然，口腔共鸣自然，咬字清楚；保持稳定统一的旁白声线，连贯耐听，不代入任何角色，不使用夸张播音腔。",
    "narrator_male": "角色：一位专业中文有声书旁白，固定为同一位成年男性。音色沉稳、温暖、饱满，声线中低沉但清晰，胸腔共鸣自然，气息稳定，咬字清楚，声线连贯耐听；不代入任何角色，不使用夸张播音腔。",
    "female_adult_01": "一位二十多岁的中文女性，声音温暖且富有表现力，清晰自然。",
    "female_adult_02": "一位年轻中文女性，声线明亮清澈，咬字轻巧，富有活力。",
    "female_adult_03": "一位年轻中文女性，嗓音柔软细腻，语气温柔治愈。",
    "female_adult_04": "一位年轻中文女性，声音活泼有能量，节奏轻快，情绪鲜明。",
    "female_adult_05": "一位成熟中文女性，声线醇雅从容，表达克制而有分寸。",
    "male_adult_01": "一位四十岁左右的中年男性，低沉浑厚，富有磁性和力量感。",
    "male_adult_02": "一位三十岁左右的中文男性，声音清晰利落，咬字准确，表达干练。",
    "male_adult_03": "一位三十岁左右的中文男性，声音温暖亲切，语气自然随和。",
    "male_adult_04": "一位四十岁左右的中年男性，声线坚定威严，具有领导者的沉稳气场。",
    "male_adult_05": "一位成熟中文男性，声音平静醇厚，节奏从容，令人安心。",
    "neutral_dialogue_01": "一位成年中文说话者，声线自然中性，咬字清楚，避免夸张表演。",
}


# ---------------------------------------------------------------------------
# Optional imports — declared at module level so tests can patch them
# ---------------------------------------------------------------------------

try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
except ImportError:
    ParlerTTSForConditionalGeneration = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioArtifact:
    kind: str
    path: Path
    duration_seconds: float


# ---------------------------------------------------------------------------
# Mock backend (used in tests / offline)
# ---------------------------------------------------------------------------

class MockTTSBackend:
    backend_id = "mock"

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"
        duration = _duration_for_text(segment.get("text", ""))
        _write_silence(output_path, duration_seconds=duration)
        return AudioArtifact(
            kind="segment_audio",
            path=output_path,
            duration_seconds=duration,
        )


def _voice_context_for_segment(segment: dict) -> tuple[str, bool, str]:
    """Resolve the stable identity description shared by clone backends."""

    voice_id = segment.get("voiceId", "narrator_default")
    is_narrator = (
        str(segment.get("speakerId") or "").strip() == "narrator"
        or is_narrator_voice_id(voice_id)
    )
    if is_narrator or is_narrator_voice_id(voice_id):
        voice_id = normalize_narrator_voice_id(voice_id)
        # A per-segment direction can change delivery, never the narrator's
        # base identity. Both MiMo and VoxCPM2 use this same invariant.
        description = _MIMO_VOICE_DESIGNS[voice_id]
    else:
        description = (
            segment.get("voiceDesign")
            or segment.get("voiceDescription")
            or _MIMO_VOICE_DESIGNS.get(
                voice_id,
                _MIMO_VOICE_DESIGNS[DEFAULT_NARRATOR_VOICE_ID],
            )
        )
    return str(voice_id), is_narrator, str(description).strip()


def voxcpm2_language_for_segment(segment: dict) -> str:
    """Resolve the language used by VoxCPM2's prompt controls."""

    return resolve_text_language(
        str(segment.get("text") or ""),
        str(segment.get("language") or "").strip() or None,
    )


def _bounded_prompt_text(value: object, limit: int) -> str:
    """Normalize a prompt fragment and trim it at a readable boundary."""

    normalized = " ".join(str(value or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    candidate = normalized[:limit]
    boundary = max(
        candidate.rfind(mark)
        for mark in ("。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ")
    )
    if boundary >= max(20, int(limit * 0.55)):
        candidate = candidate[:boundary]
    return candidate.rstrip(" ，,；;:：")


_VOXCPM2_ENGLISH_PROFILE_HINTS: tuple[tuple[str, str], ...] = (
    ("专业中文有声书旁白", "professional audiobook narrator"),
    ("有声书旁白", "audiobook narrator"),
    ("固定为同一位成年女性", "consistent adult female voice"),
    ("固定为同一位成年男性", "consistent adult male voice"),
    ("成年女性", "adult female voice"),
    ("成年男性", "adult male voice"),
    ("年轻女性", "young female voice"),
    ("年轻男性", "young male voice"),
    ("成熟女性", "mature female voice"),
    ("成熟男性", "mature male voice"),
    ("中年女性", "middle-aged female voice"),
    ("中年男性", "middle-aged male voice"),
    ("女性", "female voice"),
    ("男性", "male voice"),
    ("洪亮", "projecting"),
    ("饱满", "full"),
    ("温暖", "warm"),
    ("柔和", "soft"),
    ("清晰", "clear"),
    ("清亮", "bright"),
    ("明亮", "bright"),
    ("低沉", "low-pitched"),
    ("浑厚", "rich"),
    ("醇厚", "smooth and rich"),
    ("沉稳", "steady"),
    ("坚定", "firm"),
    ("威严", "authoritative"),
    ("柔软", "soft"),
    ("细腻", "delicate"),
    ("咬字清楚", "clear diction"),
    ("咬字利落", "crisp diction"),
    ("气息稳定", "steady breath"),
    ("胸腔共鸣自然", "natural chest resonance"),
    ("声线连贯耐听", "consistent and listenable delivery"),
)


def _english_profile_control_from_chinese(description: str) -> str:
    """Make a small deterministic English fallback for Chinese role designs.

    The canonical design remains untouched.  This fallback only keeps the
    audible identity anchors that can be translated safely without adding a
    second model call.
    """

    hints: list[str] = []
    for source, target in _VOXCPM2_ENGLISH_PROFILE_HINTS:
        if target in {
            "audiobook narrator",
            "adult female voice",
            "adult male voice",
            "female voice",
            "male voice",
        } and any(target in existing for existing in hints):
            continue
        if source in description and target not in hints:
            hints.append(target)
    if not hints:
        hints.append("natural clear diction")
    if not any("audiobook narrator" in hint for hint in hints):
        hints.insert(0, "stable audiobook voice")
    else:
        hints.insert(0, "stable voice")
    return ", ".join(hints)


def _voxcpm2_language_key(language: object) -> str:
    return "zh" if str(language or "").strip().lower().split("-", 1)[0] == "zh" else "en"


def voxcpm2_profile_control(description: object, language: str = "zh") -> str:
    """Project the canonical role design into stable VoxCPM2 syntax."""

    normalized = " ".join(str(description or "").split()).strip()
    normalized = re.sub(
        r"^(?:角色|role|voice)\s*[:：-]\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    if _voxcpm2_language_key(language) != "zh":
        if re.search(r"[\u3400-\u9fff]", normalized):
            normalized = _english_profile_control_from_chinese(normalized)
    return _bounded_prompt_text(
        normalized,
        _VOXCPM2_PROFILE_CONTROL_MAX_CHARACTERS,
    )


def voxcpm2_reference_text(language: str) -> str:
    """Return the fixed neutral sentence used to create a local profile."""

    return _VOXCPM2_REFERENCE_TEXTS[_voxcpm2_language_key(language)]


# ---------------------------------------------------------------------------
# Xiaomi MiMo TTS backend (cloud voice-design / voice-clone models)
# ---------------------------------------------------------------------------

class MiMoTTSBackend:
    backend_id = "mimo"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = _MIMO_MODEL_ID,
        request_audio=None,
        key_loader=None,
        voice_profile_directory: Path | str | None = None,
        reference_model_id: str = _MIMO_VOICE_DESIGN_MODEL_ID,
    ) -> None:
        loader = key_loader or _load_mimo_api_key
        self._api_key = api_key or os.environ.get("MIMO_API_KEY") or loader()
        if not self._api_key:
            raise RuntimeError(
                "MIMO_API_KEY is not configured. Store it in the macOS Keychain "
                f"service '{_MIMO_KEYCHAIN_SERVICE}' or set MIMO_API_KEY."
            )
        self._model_id = model_id
        self._request_audio = request_audio or self._request_audio_from_api
        self._uses_default_request_audio = request_audio is None
        self._voice_profile_directory = (
            Path(voice_profile_directory) if voice_profile_directory else None
        )
        self._reference_model_id = reference_model_id
        self._reference_audio_cache: dict[str, str] = {}
        # Profile creation writes both WAV and JSON sidecar files. Keep the
        # whole check/create/replace sequence atomic when segment synthesis is
        # later dispatched across threads.
        self._voice_profile_lock = threading.RLock()

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"
        voice_sample: str | None = None
        if self._model_id == _MIMO_VOICE_CLONE_MODEL_ID:
            profile_directory = self._voice_profile_directory or directory / ".voice-profiles"
            voice_sample = self._ensure_voice_sample(segment, profile_directory)
        request = self._build_request(segment, voice_sample)

        def validator(encoded_audio: str) -> tuple[bytes, float]:
            return _decode_and_validate_mimo_segment_wav(encoded_audio, segment)

        if self._uses_default_request_audio:
            audio_bytes, duration = self._request_audio_from_api(
                request,
                response_validator=validator,
            )
        else:
            audio_bytes, duration = validator(self._request_audio(request))

        # A rejected provider response must never overwrite the prior accepted
        # cache entry. Only atomically replace the path after validation passes.
        temporary_path = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary_path.write_bytes(audio_bytes)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return AudioArtifact("segment_audio", output_path, duration)

    def prepare_voice_profiles(
        self,
        segments: list[dict],
        profile_directory: Path | str | None = None,
    ) -> None:
        """Create/read all required clone profiles before TTS threads start.

        Reference generation is intentionally serialized. Besides avoiding
        duplicate provider requests, this prevents two threads from replacing
        the same WAV/metadata pair while the other thread is reading it.
        """
        if self._model_id != _MIMO_VOICE_CLONE_MODEL_ID:
            return
        directory = Path(profile_directory) if profile_directory else self._voice_profile_directory
        if directory is None:
            directory = Path(".voice-profiles")
        seen_signatures: set[str] = set()
        for segment in segments:
            voice_id, _, description = self._voice_context(segment)
            signature = _voice_profile_signature(
                voice_id=voice_id,
                description=description,
                reference_model_id=self._reference_model_id,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            self._ensure_voice_sample(segment, directory)

    def _voice_context(self, segment: dict) -> tuple[str, bool, str]:
        return _voice_context_for_segment(segment)

    def _build_request(self, segment: dict, voice_sample: str | None = None) -> dict:
        _, is_narrator, description = self._voice_context(segment)
        emotion = _MIMO_STYLE_DIRECTIONS.get(
            segment.get("emotion", "neutral"), _MIMO_STYLE_DIRECTIONS["neutral"]
        )
        pace_id = segment.get("pace", "normal")
        pace = _MIMO_PACE_DIRECTIONS.get(pace_id, _MIMO_PACE_DIRECTIONS["normal"])
        direction = str(segment.get("voiceDirection") or "").strip()
        if not direction:
            direction = f"{emotion}；{pace}"
        if is_narrator:
            direction = (
                f"{direction}；保持当前旁白的固定性别、年龄、音高和基础音色，"
                "不要模仿角色，不要因情绪或语速改变成另一种声音。"
            )
        scene = str(
            segment.get("voiceSceneContext")
            or segment.get("sceneContext")
            or "当前片段的叙事或对白场景。"
        ).strip()
        fixed_design = str(description).strip()
        if not fixed_design.startswith(("角色：", "角色:")):
            fixed_design = f"角色：{fixed_design}"
        if self._model_id == _MIMO_VOICE_CLONE_MODEL_ID:
            if not voice_sample:
                raise RuntimeError("MiMo voiceclone requires a reusable voice sample.")
            return {
                "model": self._model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "角色：严格复用参考音频中的同一位说话者。保持参考音频的固定性别、"
                            "年龄、音高、基础音色、共鸣位置、气息质感和咬字基线；不得因场景、"
                            "情绪、语速或角色扮演改变为另一种声音。\n\n"
                            f"基础声线设计（只用于身份约束，不重新设计声音）：{fixed_design}\n\n"
                            f"场景：{scene}\n\n"
                            f"指导：{direction}"
                        ),
                    },
                    {"role": "assistant", "content": segment["text"]},
                ],
                "audio": {"format": "wav", "voice": voice_sample},
            }
        return {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{fixed_design}\n\n"
                        f"场景：{scene}\n\n"
                        f"指导：{direction}"
                    ),
                },
                {"role": "assistant", "content": segment["text"]},
            ],
            "audio": {"format": "wav", "optimize_text_preview": False},
        }

    def _ensure_voice_sample(self, segment: dict, profile_directory: Path) -> str:
        with self._voice_profile_lock:
            voice_id, is_narrator, description = self._voice_context(segment)
            profile_directory.mkdir(parents=True, exist_ok=True)
            profile_path = profile_directory / f"{_safe_voice_profile_name(voice_id)}.wav"
            metadata_path = profile_path.with_suffix(".json")
            lock_path = profile_path.with_suffix(".lock")
            signature = _voice_profile_signature(
                voice_id=voice_id,
                description=description,
                reference_model_id=self._reference_model_id,
            )

            cached = self._reference_audio_cache.get(signature)
            if cached and _is_readable_wav(profile_path):
                return cached
            # The in-process RLock above does not help when multiple chapter
            # subprocesses share one book's voice-profiles directory. fcntl
            # keeps the complete cache-check/design/replace sequence exclusive
            # across those processes without serialising their ordinary TTS.
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                    if (
                        metadata.get("signature") == signature
                        and metadata.get("version") == _MIMO_VOICE_PROFILE_VERSION
                        and _is_readable_wav(profile_path)
                    ):
                        data_uri = _audio_data_uri(profile_path.read_bytes())
                        self._reference_audio_cache[signature] = data_uri
                        return data_uri

                    # This request deliberately excludes the current segment's
                    # scene, emotion, and pace. It creates the stable identity
                    # anchor reused by every chapter and every later segment.
                    fixed_design = description
                    if not fixed_design.startswith(("角色：", "角色:")):
                        fixed_design = f"角色：{fixed_design}"
                    reference_request = {
                        "model": self._reference_model_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"{fixed_design}\n\n"
                                    "指导：生成一段自然、稳定、克制的基础音色参考样本。"
                                    "只建立固定的性别、年龄、音高、音色、共鸣、气息和咬字基线；"
                                    "不要加入临时场景、明显情绪、角色模仿、夸张表演或后期效果。"
                                ),
                            },
                            {"role": "assistant", "content": _MIMO_REFERENCE_TEXT},
                        ],
                        "audio": {"format": "wav", "optimize_text_preview": False},
                    }
                    encoded_reference = self._request_audio(reference_request)
                    reference_bytes = _decode_mimo_wav(encoded_reference, "MiMo voice design")
                    # Validate before replacing an existing profile so a failed
                    # refresh never destroys the last known-good identity anchor.
                    temporary_path = profile_path.with_name(
                        f".{profile_path.name}.{os.getpid()}.tmp"
                    )
                    try:
                        temporary_path.write_bytes(reference_bytes)
                        if not _is_readable_wav(temporary_path):
                            raise RuntimeError(
                                "MiMo voice design returned an unreadable reference WAV."
                            )
                        temporary_path.replace(profile_path)
                        metadata_path.write_text(
                            json.dumps(
                                {
                                    "version": _MIMO_VOICE_PROFILE_VERSION,
                                    "signature": signature,
                                    "voiceId": voice_id,
                                    "voiceDesign": description,
                                    "referenceModel": self._reference_model_id,
                                    "referenceText": _MIMO_REFERENCE_TEXT,
                                    "isNarrator": is_narrator,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    finally:
                        temporary_path.unlink(missing_ok=True)

                    data_uri = _audio_data_uri(reference_bytes)
                    self._reference_audio_cache[signature] = data_uri
                    return data_uri
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _request_audio_from_api(self, payload: dict, response_validator=None):
        encoded_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        max_attempts = _mimo_max_attempts()
        last_error: MiMoRequestError | None = None
        # Keep the file lock for the entire retry cycle. This is deliberately
        # stricter than only serializing individual HTTP attempts: a 429 must
        # not let another queued audiobook request create a fresh burst while
        # the provider asked this one to cool down.
        with _mimo_request_rate_gate() as rate_gate:
            for attempt in range(1, max_attempts + 1):
                rate_gate.wait_for_turn()
                request = urllib.request.Request(
                    _MIMO_ENDPOINT,
                    data=encoded_payload,
                    headers={
                        "api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                retry_delay = 0.0
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        result = json.loads(response.read().decode("utf-8"))
                    try:
                        encoded_audio = result["choices"][0]["message"]["audio"]["data"]
                    except (KeyError, IndexError, TypeError) as error:
                        raise RuntimeError(
                            "MiMo response did not contain choices[0].message.audio.data."
                        ) from error
                    if response_validator is not None:
                        return response_validator(encoded_audio)
                    return encoded_audio
                except MiMoRequestError as error:
                    last_error = error
                    if not last_error.retryable or attempt >= max_attempts:
                        raise
                    retry_delay = _mimo_retry_delay_seconds(attempt)
                except urllib.error.HTTPError as error:
                    detail = error.read().decode("utf-8", errors="replace")[:500]
                    retry_after = _retry_after_seconds(
                        error.headers.get("Retry-After") if error.headers else None
                    )
                    rate_limited = error.code == 429
                    retry_delay = _mimo_retry_delay_seconds(
                        attempt,
                        retry_after=retry_after,
                        rate_limited=rate_limited,
                    )
                    if rate_limited:
                        rate_gate.set_cooldown(retry_delay)
                    last_error = MiMoRequestError(
                        f"MiMo API request failed with HTTP {error.code}: {detail}",
                        retryable=error.code in {408, 425, 429} or error.code >= 500,
                    )
                    if not last_error.retryable or attempt >= max_attempts:
                        raise last_error from error
                except (urllib.error.URLError, TimeoutError) as error:
                    last_error = MiMoRequestError(
                        f"MiMo API request failed: {error}", retryable=True
                    )
                    if attempt >= max_attempts:
                        raise last_error from error
                    retry_delay = _mimo_retry_delay_seconds(attempt)

                if retry_delay > 0:
                    time.sleep(retry_delay)

        raise last_error or RuntimeError("MiMo API request failed.")


# ---------------------------------------------------------------------------
# VoxCPM2 backend (isolated local model process)
# ---------------------------------------------------------------------------

class VoxCPM2TTSBackend:
    """Adapter for the locally installed VoxCPM2 reference-cloning workflow.

    The model lives in ``data/voxcpm2/.venv`` and must never be imported by
    this worker interpreter.  A chapter passes all uncached segments to the
    runner together, allowing the isolated process to load the 2B model once.
    """

    backend_id = "voxcpm2"

    def __init__(
        self,
        model_id: str = VOXCPM2_MODEL_ID,
        *,
        voice_profile_directory: Path | str | None = None,
        runner_python: Path | str | None = None,
        model_path: Path | str | None = None,
        runner_path: Path | str | None = None,
    ) -> None:
        paths = voxcpm2_paths()
        self._model_id = str(model_id or VOXCPM2_MODEL_ID)
        self._runner_python = Path(runner_python) if runner_python else paths["python"]
        self._model_path = Path(model_path) if model_path else paths["model"]
        self._runner_path = (
            Path(runner_path)
            if runner_path
            else Path(__file__).with_name("voxcpm2_runner.py")
        )
        self._voice_profile_directory = (
            Path(voice_profile_directory) if voice_profile_directory else None
        )
        self._device: str | None = None

    def synthesize_segment(
        self,
        segment: dict,
        output_directory: Path | str,
    ) -> AudioArtifact:
        return self.synthesize_segments([segment], output_directory)[0]

    def synthesize_segments(
        self,
        segments: list[dict],
        output_directory: Path | str,
    ) -> list[AudioArtifact]:
        """Synthesize source-order segments with one isolated model load."""

        if not segments:
            return []
        self._validate_runtime()
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        profile_directory = self._voice_profile_directory or directory / ".voice-profiles"
        profile_directory.mkdir(parents=True, exist_ok=True)

        profiles: list[dict[str, object]] = []
        profile_request_keys: set[tuple[Path, str]] = set()
        runner_segments: list[dict[str, object]] = []
        expected_paths: dict[str, Path] = {}
        for segment in segments:
            segment_id = str(segment.get("id") or "").strip()
            if not segment_id:
                raise RuntimeError("VoxCPM2 cannot synthesize a segment without an id.")
            voice_id, _, description = _voice_context_for_segment(segment)
            language = voxcpm2_language_for_segment(segment)
            profile_control = voxcpm2_profile_control(description, language)
            reference_text = voxcpm2_reference_text(language)
            profile_path = profile_directory / (
                f"{_safe_voice_profile_name(voice_id)}_{language}.wav"
            )
            metadata_path = profile_path.with_suffix(".json")
            signature = _voxcpm2_voice_profile_signature(
                voice_id=voice_id,
                description=description,
                language=language,
            )
            profile_key = (profile_path, signature)
            if (
                not _voxcpm2_profile_is_usable(
                    profile_path,
                    metadata_path,
                    signature=signature,
                )
                and profile_key not in profile_request_keys
            ):
                profile_request_keys.add(profile_key)
                profiles.append(
                    {
                        "voiceId": voice_id,
                        "profilePath": str(profile_path),
                        "metadataPath": str(metadata_path),
                        "lockPath": str(profile_path.with_suffix(".lock")),
                        "signature": signature,
                        "voiceDesign": description,
                        "profileControl": profile_control,
                        "referenceText": reference_text,
                        "language": language,
                        "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
                        "profileLoudness": voxcpm2_profile_loudness(),
                    }
                )
            output_path = directory / f"{segment_id}.wav"
            expected_paths[segment_id] = output_path
            runner_segments.append(
                {
                    "id": segment_id,
                    "text": str(segment.get("text") or ""),
                    "delivery": self._delivery_instruction(segment),
                    # VoxCPM2 also counts the parenthesized delivery control
                    # when it computes its internal generation ceiling. Pass
                    # the speech-only quality ceiling so a long control cannot
                    # turn a short utterance into an unbounded decode.
                    "maxDurationSeconds": maximum_duration_for_segment(
                        segment.get("text", ""),
                        segment.get("pace", "normal"),
                    ),
                    "language": language,
                    "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
                    "referenceWavPath": str(profile_path),
                    "outputPath": str(output_path),
                }
            )

        response = self._run_runner(
            {
                "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
                "profiles": profiles,
                "segments": runner_segments,
            }
        )
        device = response.get("device")
        self._device = str(device) if isinstance(device, str) and device else None
        raw_results = response.get("segments")
        if not isinstance(raw_results, list):
            raise RuntimeError("VoxCPM2 runner did not return segment results.")
        results_by_id = {
            str(item.get("id")): item
            for item in raw_results
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        artifacts: list[AudioArtifact] = []
        for segment in segments:
            segment_id = str(segment["id"])
            result = results_by_id.get(segment_id)
            expected_path = expected_paths[segment_id]
            if result is None:
                raise RuntimeError(f"VoxCPM2 runner did not return segment {segment_id}.")
            result_path = Path(str(result.get("path") or ""))
            if result_path.resolve() != expected_path.resolve() or not _is_readable_wav(
                expected_path
            ):
                raise RuntimeError(
                    f"VoxCPM2 runner did not create a readable WAV for segment {segment_id}."
                )
            try:
                duration = float(result.get("durationSeconds"))
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"VoxCPM2 runner returned an invalid duration for segment {segment_id}."
                ) from error
            if not math.isfinite(duration) or duration <= 0:
                raise RuntimeError(
                    f"VoxCPM2 runner returned a non-positive duration for segment {segment_id}."
                )
            artifacts.append(AudioArtifact("segment_audio", expected_path, duration))
        return artifacts

    def _validate_runtime(self) -> None:
        if self._model_id != VOXCPM2_MODEL_ID:
            raise RuntimeError(f"Unsupported VoxCPM2 model: {self._model_id}")
        if not self._runner_python.is_file():
            raise RuntimeError(
                f"VoxCPM2 isolated Python is missing: {self._runner_python}"
            )
        if not self._model_path.is_dir():
            raise RuntimeError(f"VoxCPM2 model directory is missing: {self._model_path}")
        if not self._runner_path.is_file():
            raise RuntimeError(f"VoxCPM2 runner is missing: {self._runner_path}")

    def _delivery_instruction(self, segment: dict) -> str:
        language = voxcpm2_language_for_segment(segment)
        emotion_directions = (
            _VOXCPM2_EMOTION_DIRECTIONS_ZH
            if language == "zh"
            else _VOXCPM2_EMOTION_DIRECTIONS_EN
        )
        pace_directions = (
            _VOXCPM2_PACE_DIRECTIONS_ZH
            if language == "zh"
            else _VOXCPM2_PACE_DIRECTIONS_EN
        )
        emotion = emotion_directions.get(
            str(segment.get("emotion") or "neutral").strip().lower(),
            emotion_directions["neutral"],
        )
        pace = pace_directions.get(
            str(segment.get("pace") or "normal").strip().lower(),
            pace_directions["normal"],
        )
        direction = _bounded_prompt_text(
            segment.get("voiceDirection"),
            _VOXCPM2_DIRECTION_MAX_CHARACTERS,
        )
        parts = [emotion, pace]
        if direction:
            parts.append(direction)
        separator = "，" if language == "zh" else ", "
        return separator.join(parts)

    def _run_runner(self, payload: dict[str, object]) -> dict[str, object]:
        request = {
            "modelPath": str(self._model_path),
            "device": str(os.environ.get("AUDIOBOOK_VOXCPM2_DEVICE", "auto")).strip()
            or "auto",
            **payload,
            "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
        }
        try:
            configured_timeout = int(
                os.environ.get(
                    "AUDIOBOOK_VOXCPM2_RUNNER_TIMEOUT_SECONDS",
                    str(_VOXCPM2_RUNNER_TIMEOUT_SECONDS),
                )
            )
        except ValueError:
            configured_timeout = _VOXCPM2_RUNNER_TIMEOUT_SECONDS
        timeout = max(60, min(_VOXCPM2_RUNNER_TIMEOUT_SECONDS, configured_timeout))
        with tempfile.TemporaryDirectory(prefix="audiobook-voxcpm2-") as temporary_directory:
            temporary = Path(temporary_directory)
            input_path = temporary / "input.json"
            output_path = temporary / "output.json"
            input_path.write_text(
                json.dumps(request, ensure_ascii=False), encoding="utf-8"
            )
            try:
                completed = subprocess.run(
                    [
                        str(self._runner_python),
                        str(self._runner_path),
                        str(input_path),
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"VoxCPM2 chapter synthesis timed out after {timeout} seconds."
                ) from error
            try:
                response = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                response = None
            if isinstance(response, dict) and response.get("status") == "succeeded":
                return response
            if isinstance(response, dict):
                error = response.get("error")
                if isinstance(error, dict) and isinstance(error.get("message"), str):
                    raise RuntimeError(f"VoxCPM2 synthesis failed: {error['message']}")
            detail = (completed.stderr or completed.stdout or "").strip()
            if detail:
                detail = detail.splitlines()[-1]
            raise RuntimeError(
                "VoxCPM2 runner failed"
                + (f": {detail}" if detail else f" (exit {completed.returncode})")
            )


def _safe_voice_profile_name(voice_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(voice_id)).strip("._")
    return normalized or "voice"


def _voice_profile_signature(
    *,
    voice_id: str,
    description: str,
    reference_model_id: str,
) -> str:
    payload = {
        "version": _MIMO_VOICE_PROFILE_VERSION,
        "voiceId": voice_id,
        "voiceDesign": description,
        "referenceModel": reference_model_id,
        "referenceText": _MIMO_REFERENCE_TEXT,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _voxcpm2_voice_profile_signature(
    *,
    voice_id: str,
    description: str,
    language: str = "zh",
) -> str:
    """Return the identity-cache key for a VoxCPM2 reference WAV.

    MiMo references are generated by a cloud voice-design model, whereas
    VoxCPM2 creates a local reference WAV.  Keeping the version and backend
    in a separate signature prevents either profile format from being reused
    after a backend switch, even when the voice id happens to match.
    """

    payload = {
        "version": _VOXCPM2_VOICE_PROFILE_VERSION,
        "backend": "voxcpm2",
        "modelId": VOXCPM2_MODEL_ID,
        "voiceId": voice_id,
        "voiceDescription": description,
        "voiceDesign": description,
        "profileControl": voxcpm2_profile_control(description, language),
        "language": _voxcpm2_language_key(language),
        "referenceText": voxcpm2_reference_text(language),
        "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
        "profileLoudness": voxcpm2_profile_loudness(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _voxcpm2_profile_is_usable(
    profile_path: Path,
    metadata_path: Path,
    *,
    signature: str,
) -> bool:
    """Return whether a local reference WAV matches the current backend key."""

    if not _is_readable_wav(profile_path):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("version") == _VOXCPM2_VOICE_PROFILE_VERSION
        and metadata.get("backend") == "voxcpm2"
        and metadata.get("modelId") == VOXCPM2_MODEL_ID
        and metadata.get("promptFormatVersion") == VOXCPM2_PROMPT_FORMAT_VERSION
        and profile_loudness_is_current(metadata.get("profileLoudness"))
        and metadata.get("signature") == signature
    )


def _decode_mimo_wav(encoded_audio: str, source: str) -> bytes:
    try:
        audio_bytes = base64.b64decode(encoded_audio, validate=True)
    except (ValueError, TypeError) as error:
        raise RuntimeError(f"{source} returned invalid Base64 audio data.") from error
    if not audio_bytes.startswith(b"RIFF"):
        raise RuntimeError(f"{source} response is not a WAV audio file.")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnframes() <= 0 or wav_file.getframerate() <= 0:
                raise RuntimeError(f"{source} returned an empty WAV audio file.")
    except (wave.Error, EOFError, ZeroDivisionError) as error:
        raise RuntimeError(f"{source} returned an unreadable WAV audio file.") from error
    return audio_bytes


def _decode_and_validate_mimo_segment_wav(
    encoded_audio: str,
    segment: dict,
) -> tuple[bytes, float]:
    """Decode a provider response and convert bad speech audio into a retry."""
    try:
        audio_bytes = _decode_mimo_wav(encoded_audio, "MiMo")
        quality = validate_tts_segment_wav(
            audio_bytes,
            text=segment.get("text", ""),
            pace=segment.get("pace", "normal"),
        )
    except TtsSegmentAudioQualityError as error:
        raise MiMoRequestError(
            f"MiMo returned an unusable TTS segment WAV: {error}",
            retryable=True,
            quality=error.result,
        ) from error
    except RuntimeError as error:
        raise MiMoRequestError(
            f"MiMo returned an unusable TTS segment WAV: {error}",
            retryable=True,
        ) from error
    return audio_bytes, quality.duration_seconds


def _audio_data_uri(audio_bytes: bytes) -> str:
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    data_uri = f"data:audio/wav;base64,{encoded}"
    if len(encoded) > _MIMO_MAX_REFERENCE_BASE64_LENGTH:
        raise RuntimeError(
            "MiMo voice reference audio exceeds the official 10 MB Base64 limit."
        )
    return data_uri


def _is_readable_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except (OSError, wave.Error, ZeroDivisionError):
        return False


_MIMO_STYLE_DIRECTIONS = {
    "neutral": "情绪自然克制，不带明显起伏",
    "happy": "语气真诚温暖，带着发自内心的愉悦，声线略微上扬，不要夸张",
    "sad": "情绪低沉哀伤，声线带着压抑，语句尾音微微下垂，克制而不表演",
    "angry": "带着压抑或爆发的愤怒，气息有力，咬字坚定，不可轻描淡写",
    "afraid": "声音紧绷，带着轻微颤抖，气息不稳，透出恐惧与不安",
    "tense": "语气谨慎克制，暗含压迫感，每字落地有重量，像随时准备应对变故",
    "teasing": "语气戏谑嘲弄，带着轻蔑笑意和挑衅，绝不要读成真诚的开心或轻描淡写",
    "whispering": "声音压低至耳语，气息感明显，像在秘密传递消息或低声安慰，不要使用正常讲话音量",
    "excited": "情绪高涨兴奋，语速略快，难掩内心激动，声线带着能量感",
    "tired": "声音略带疲惫，气息稍显沉重，语调偏低，像身心俱疲后勉强开口",
    "grief": "深切悲恸，声线哽咽，带着难以抑制的哭腔或泣不成声，比普通悲伤更撕裂更沉重",
    "cold": "语气冷淡疏离，毫无情感起伏，像隔着一堵玻璃说话，刻意保持距离，不带任何温度",
    "pleading": "语气恳切哀求，带着卑微或迫切，声线略低，透出无助与急切的请求",
    "surprised": "语调突然上扬，带着真实的错愕和意外，不是愉快的惊喜，而是震惊或难以置信",
    "gentle": "声音轻柔温存，语速缓慢，像在安慰受伤的人或对幼小者说话，充满体贴与关怀",
    "resolute": "语气坚定有力，咬字清晰，不可动摇，带着下定决心后的沉稳与分量",
    "nervous": "声音略带颤抖和忐忑，停顿偏多，透出内心的慌乱和不安，不同于外部压力导致的tense",
    "contemptuous": "语气冷蔑鄙视，不带笑意，居高临下，比teasing更冷、更刻薄凌厉",
    "solemn": "语气庄严肃穆，语速稳重，带着郑重与分量，适合宣誓、重要告知或庄重场合",
    "bitter": "语气苦涩愤慨，带着委屈和压抑的不平，不是公开的愤怒，而是咽下去的辛酸和不甘",
}

_MIMO_PACE_DIRECTIONS = {
    "slow": "语速舒缓，停顿自然",
    "normal": "语速适中",
    "fast": "语速偏快，节奏紧凑",
}

_VOXCPM2_EMOTION_DIRECTIONS_ZH = {
    "neutral": "情绪自然克制",
    "happy": "真诚温暖，带轻微愉悦",
    "sad": "低沉哀伤，克制收束",
    "angry": "压抑而有力的愤怒",
    "afraid": "紧绷不安，带轻微颤抖",
    "tense": "谨慎克制，暗含压迫感",
    "teasing": "带戏谑和轻微嘲弄",
    "whispering": "压低为清晰耳语，保留可懂度",
    "excited": "情绪高涨而不夸张",
    "tired": "略显疲惫，气息偏重",
    "grief": "深切悲恸，但不要失控尖叫",
    "cold": "冷淡疏离，保持克制",
    "pleading": "恳切急迫，带无助感",
    "surprised": "真实错愕，语调短促上扬",
    "gentle": "轻柔温存，带安抚感",
    "resolute": "坚定有力，落字清晰",
    "nervous": "忐忑不安，停顿略多",
    "contemptuous": "冷蔑疏离，不带笑意",
    "solemn": "庄严肃穆，稳重郑重",
    "bitter": "苦涩压抑，带不甘",
}

_VOXCPM2_PACE_DIRECTIONS_ZH = {
    "slow": "语速舒缓，保留自然停顿",
    "normal": "语速自然适中",
    "fast": "语速偏快但字音清楚",
}

_VOXCPM2_EMOTION_DIRECTIONS_EN = {
    "neutral": "natural and restrained",
    "happy": "warm and lightly cheerful",
    "sad": "subdued and sorrowful",
    "angry": "controlled but forceful anger",
    "afraid": "tense and slightly trembling",
    "tense": "guarded and quietly pressured",
    "teasing": "playfully mocking, with a sly edge",
    "whispering": "a clear, intimate whisper",
    "excited": "energized but controlled",
    "tired": "weary, with heavier breath",
    "grief": "deep grief without losing control",
    "cold": "detached and emotionally cold",
    "pleading": "urgent and vulnerable",
    "surprised": "genuinely startled, briefly rising",
    "gentle": "soft, tender, and comforting",
    "resolute": "firm and unwavering",
    "nervous": "uneasy, with slightly longer pauses",
    "contemptuous": "coldly contemptuous, without humor",
    "solemn": "grave, measured, and ceremonial",
    "bitter": "quietly bitter and resentful",
}

_VOXCPM2_PACE_DIRECTIONS_EN = {
    "slow": "slow and measured",
    "normal": "natural conversational pace",
    "fast": "quick but clearly articulated",
}

# Keep the old private names as aliases for callers that imported them during
# the initial local-backend rollout. New code selects the language explicitly.
_VOXCPM2_EMOTION_DIRECTIONS = _VOXCPM2_EMOTION_DIRECTIONS_ZH
_VOXCPM2_PACE_DIRECTIONS = _VOXCPM2_PACE_DIRECTIONS_ZH


def _load_mimo_api_key() -> str | None:
    security = Path("/usr/bin/security")
    if not security.exists():
        return None
    completed = subprocess.run(
        [
            str(security),
            "find-generic-password",
            "-s",
            _MIMO_KEYCHAIN_SERVICE,
            "-w",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


# ---------------------------------------------------------------------------
# Kokoro TTS backend (primary)
# ---------------------------------------------------------------------------

class KokoroTTSBackend:
    backend_id = "kokoro"

    def __init__(self, lang_code: str = "a") -> None:
        self._lang_code = lang_code
        self._pipeline = None

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        import numpy as np
        import soundfile as sf

        self._ensure_pipeline()

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"

        if str(segment.get("speakerId") or "").strip() == "narrator":
            voice_id = normalize_narrator_voice_id(segment.get("voiceId"))
        else:
            voice_id = segment.get("fallbackVoiceId") or segment.get(
                "voiceId", DEFAULT_NARRATOR_VOICE_ID
            )
        voice_name = _kokoro_voice_for(voice_id)
        text = segment["text"]

        generator = self._pipeline(text, voice=voice_name, speed=1.0, split_pattern=None)

        segments_audio = []
        for result in generator:
            audio_segment = result.audio
            if hasattr(audio_segment, "cpu"):
                audio_segment = audio_segment.cpu().numpy().squeeze()
            else:
                audio_segment = np.array(audio_segment).squeeze()
            if audio_segment.ndim == 1 and len(audio_segment) > 0:
                segments_audio.append(audio_segment)

        audio = np.concatenate(segments_audio) if segments_audio else np.zeros(0, dtype=np.float32)
        sf.write(str(output_path), audio, 24000)

        duration = len(audio) / 24000
        return AudioArtifact(
            kind="segment_audio",
            path=output_path,
            duration_seconds=duration,
        )

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        import torch

        requested_device = os.environ.get("AUDIOBOOK_TTS_DEVICE", "auto")
        kokoro_device = _select_kokoro_device(torch, requested_device)

        # KPipeline only natively supports 'cpu'/'cuda', so init on CPU first
        init_device = kokoro_device if kokoro_device in ("cpu", "cuda") else "cpu"
        self._pipeline = KPipeline(lang_code=self._lang_code, device=init_device)

        # Move model to MPS after init if available
        if kokoro_device == "mps" and self._pipeline.model is not None:
            self._pipeline.model.to("mps")
            self._pipeline.model.eval()
            self._device = "mps"
        else:
            self._device = kokoro_device


# ---------------------------------------------------------------------------
# Parler TTS backend (secondary, kept for comparison / voice-description mode)
# ---------------------------------------------------------------------------

class ParlerTTSBackend:
    backend_id = "parler"

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        import soundfile as sf

        self._ensure_model()

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"

        description = self._build_description(segment)
        text = segment["text"]

        audio_array = self._generate(description, text)
        if audio_array.dtype.name == "float16":
            audio_array = audio_array.astype("float32")

        sf.write(str(output_path), audio_array, self._model.config.sampling_rate)

        duration = len(audio_array) / self._model.config.sampling_rate
        return AudioArtifact(
            kind="segment_audio",
            path=output_path,
            duration_seconds=duration,
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch

        requested_device = os.environ.get("AUDIOBOOK_TTS_DEVICE", "auto")
        self._device = _select_torch_device(torch, requested_device)

        dtype = torch.float16 if self._device in ("mps", "cuda") else torch.float32
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self._model_id, torch_dtype=dtype
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

    def _build_description(self, segment: dict) -> str:
        voice_id = segment.get("voiceId", "narrator_default")
        is_narrator = (
            str(segment.get("speakerId") or "").strip() == "narrator"
            or is_narrator_voice_id(voice_id)
        )
        if is_narrator:
            voice_id = normalize_narrator_voice_id(voice_id)
        voice_entry = VOICE_REGISTRY.get(voice_id, VOICE_REGISTRY[DEFAULT_NARRATOR_VOICE_ID])
        base = (
            voice_entry.get("parlerDescription", "A clear speaker.")
            if is_narrator
            else segment.get("voiceDesign") or segment.get("voiceDescription") or voice_entry.get(
                "parlerDescription", "A clear speaker."
            )
        )

        emotion = segment.get("emotion", "neutral")
        pace = segment.get("pace", "normal")

        emotion_mod = _EMOTION_MODIFIERS.get(emotion, _EMOTION_MODIFIERS["neutral"])
        pace_mod = _PACE_MODIFIERS.get(pace, "")
        direction = str(segment.get("voiceDirection") or "").strip()
        parts = [base, emotion_mod, direction]
        if pace_mod:
            parts.append(pace_mod)
        return " ".join(p for p in parts if p)

    def _generate(self, description: str, text: str):
        import torch

        desc_ids = self._tokenizer(description, return_tensors="pt").input_ids.to(self._device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        with torch.inference_mode():
            generation = self._model.generate(
                input_ids=desc_ids,
                prompt_input_ids=prompt_ids,
                do_sample=False,
            )

        return generation.cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def voice_registry() -> dict[str, dict]:
    return VOICE_REGISTRY.copy()


def voice_options(backend: str = "mimo") -> list[dict[str, str]]:
    """Return the voice IDs that the requested backend can actually use."""
    normalized_backend = str(backend or "mimo").strip().lower()
    if normalized_backend == "mimo":
        voice_ids = set(_MIMO_VOICE_DESIGNS)
    else:
        voice_ids = set(VOICE_REGISTRY)

    return [
        {
            "id": voice_id,
            "displayName": str(entry["displayName"]),
            "backend": normalized_backend,
        }
        for voice_id, entry in VOICE_REGISTRY.items()
        if voice_id in voice_ids
    ]


def _select_torch_device(torch_module, requested_device: str = "auto") -> str:
    requested = requested_device.strip().lower()
    if requested not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError(
            "AUDIOBOOK_TTS_DEVICE must be one of: auto, mps, cuda, cpu"
        )

    mps_available = torch_module.backends.mps.is_available()
    cuda_available = torch_module.cuda.is_available()

    if requested == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested for TTS, but torch.backends.mps.is_available() is false."
            )
        return "mps"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested for TTS, but torch.cuda.is_available() is false."
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"

    if mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def _select_kokoro_device(torch_module, requested_device: str = "auto") -> str:
    """Select device for Kokoro. Returns device string for KPipeline init.
    MPS acceleration is applied post-init by moving the model manually."""
    requested = requested_device.strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    if requested in ("mps", "auto"):
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def _kokoro_voice_for(voice_id: str) -> str:
    """Map internal voice IDs to Kokoro voice names."""
    voice_entry = VOICE_REGISTRY.get(voice_id, VOICE_REGISTRY["narrator_default"])
    return voice_entry.get("kokoroVoice", "af_heart")


def _duration_for_text(text: str) -> float:
    word_count = len(text.split())
    return max(0.25, min(2.0, word_count * 0.08))


def _write_silence(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 16_000
    frame_count = math.ceil(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
