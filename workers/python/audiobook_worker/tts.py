from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from audiobook_worker.script_builder import VOICE_REGISTRY

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
_MIMO_MODEL_ID = "mimo-v2.5-tts-voicedesign"
_MIMO_KEYCHAIN_SERVICE = "audiobook-generator.mimo-api-key"

_MIMO_VOICE_DESIGNS: dict[str, str] = {
    "narrator_default": "一位三十岁左右的中文女声，温暖沉静，咬字清晰，像专业有声书演播者，叙述自然克制。",
    "narrator_female": "一位成熟的中文女声，柔和优雅，气息稳定，适合娓娓道来的长篇叙事。",
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


# ---------------------------------------------------------------------------
# Xiaomi MiMo TTS backend (cloud voice-design model)
# ---------------------------------------------------------------------------

class MiMoTTSBackend:
    backend_id = "mimo"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = _MIMO_MODEL_ID,
        request_audio=None,
        key_loader=None,
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

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"
        encoded_audio = self._request_audio(self._build_request(segment))
        try:
            audio_bytes = base64.b64decode(encoded_audio, validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError("MiMo returned invalid Base64 audio data.") from error
        if not audio_bytes.startswith(b"RIFF"):
            raise RuntimeError("MiMo response is not a WAV audio file.")
        output_path.write_bytes(audio_bytes)
        try:
            with wave.open(str(output_path), "rb") as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
        except (wave.Error, ZeroDivisionError) as error:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("MiMo returned an unreadable WAV audio file.") from error
        return AudioArtifact("segment_audio", output_path, duration)

    def _build_request(self, segment: dict) -> dict:
        voice_id = segment.get("voiceId", "narrator_default")
        description = segment.get("voiceDescription") or _MIMO_VOICE_DESIGNS.get(
            voice_id, _MIMO_VOICE_DESIGNS["narrator_default"]
        )
        emotion = _MIMO_STYLE_DIRECTIONS.get(
            segment.get("emotion", "neutral"), _MIMO_STYLE_DIRECTIONS["neutral"]
        )
        pace_id = "normal" if segment.get("speakerId") == "narrator" else segment.get(
            "pace", "normal"
        )
        pace = _MIMO_PACE_DIRECTIONS.get(pace_id, _MIMO_PACE_DIRECTIONS["normal"])
        return {
            "model": self._model_id,
            "messages": [
                {"role": "user", "content": f"{description}{emotion}，{pace}。"},
                {"role": "assistant", "content": segment["text"]},
            ],
            "audio": {"format": "wav", "optimize_text_preview": False},
        }

    def _request_audio_from_api(self, payload: dict) -> str:
        request = urllib.request.Request(
            _MIMO_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MiMo API request failed with HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"MiMo API request failed: {error}") from error
        try:
            return result["choices"][0]["message"]["audio"]["data"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("MiMo response did not contain choices[0].message.audio.data.") from error


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

        voice_id = segment.get("fallbackVoiceId") or segment.get(
            "voiceId", "narrator_default"
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
        voice_entry = VOICE_REGISTRY.get(voice_id, VOICE_REGISTRY["narrator_default"])
        base = segment.get("voiceDescription") or voice_entry.get(
            "parlerDescription", "A clear speaker."
        )

        emotion = segment.get("emotion", "neutral")
        pace = segment.get("pace", "normal")

        emotion_mod = _EMOTION_MODIFIERS.get(emotion, _EMOTION_MODIFIERS["neutral"])
        pace_mod = _PACE_MODIFIERS.get(pace, "")

        parts = [base, emotion_mod]
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
