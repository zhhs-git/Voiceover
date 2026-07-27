# Parler TTS Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a `ParlerTTSBackend` that generates real speech from audiobook script segments using Parler TTS on Apple MPS (GPU), then run an end-to-end test against a real book.

**Architecture:** `ParlerTTSBackend` follows the same `synthesize_segment(segment, output_dir) → AudioArtifact` interface as `MockTTSBackend`. Each voice registry entry gains a `parlerDescription` base string; the backend appends emotion/pace modifiers at synthesis time — Parler's core superpower. Model lazy-loads on first call using MPS device when available, falling back to CPU.

**Tech Stack:** `parler-tts` (HuggingFace), `transformers`, `torch` (MPS), `soundfile`, existing `audiobook_worker` pipeline, Project Gutenberg EPUB for test book.

**Working directory:** `.worktrees/mvp-implementation/workers/python`

---

### Task 1: Install dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add parler-tts dependencies to pyproject.toml**

Add under `[project.dependencies]`:
```toml
dependencies = [
  "beautifulsoup4>=4.12.0",
  "ebooklib>=0.18",
  "pymupdf>=1.24.0",
  "torch>=2.3.0",
  "transformers>=4.40.0",
  "parler-tts>=1.0.0",
  "soundfile>=0.12.1",
]
```

**Step 2: Install into the existing venv**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pip install torch transformers "parler-tts>=1.0.0" soundfile
```

Expected: packages install successfully. `parler-tts` pulls in `transformers` and `torch` automatically.

**Step 3: Verify MPS is available**

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected: `True` on Apple Silicon Mac.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add parler-tts, torch, soundfile dependencies"
```

---

### Task 2: Add parlerDescription to voice registry

**Files:**
- Modify: `audiobook_worker/script_builder.py:7-48`

**Step 1: Write the failing test**

In `tests/test_tts.py`, add:

```python
def test_voice_registry_has_parler_descriptions():
    voices = voice_registry()
    for voice_id, entry in voices.items():
        assert "parlerDescription" in entry, f"{voice_id} missing parlerDescription"
        assert len(entry["parlerDescription"]) > 20, f"{voice_id} description too short"
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_tts.py::test_voice_registry_has_parler_descriptions -v
```

Expected: FAIL — `AssertionError: narrator_default missing parlerDescription`

**Step 3: Add parlerDescription to each voice entry in VOICE_REGISTRY**

Replace the `VOICE_REGISTRY` dict in `audiobook_worker/script_builder.py` with:

```python
VOICE_REGISTRY = {
    "narrator_default": {
        "id": "narrator_default",
        "displayName": "Default Narrator",
        "genderPresentation": "neutral",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "parler",
        "licenseNotes": "Parler TTS Apache 2.0",
        "parlerDescription": (
            "A middle-aged male speaker with a warm, clear, and measured voice "
            "delivers the narration at a comfortable pace in a quiet studio environment. "
            "The recording is clean with no background noise."
        ),
    },
    "female_adult_01": {
        "id": "female_adult_01",
        "displayName": "Female Adult 01",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "afraid", "happy", "sad", "angry", "excited"],
        "backend": "parler",
        "licenseNotes": "Parler TTS Apache 2.0",
        "parlerDescription": (
            "A young adult female speaker with a clear, expressive voice "
            "delivers her lines in a quiet indoor setting. "
            "The recording is crisp with no background noise."
        ),
    },
    "male_adult_01": {
        "id": "male_adult_01",
        "displayName": "Male Adult 01",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "angry", "tense", "excited"],
        "backend": "parler",
        "licenseNotes": "Parler TTS Apache 2.0",
        "parlerDescription": (
            "A adult male speaker with a deep, resonant voice "
            "delivers his lines in a quiet indoor setting. "
            "The recording is clean with no background noise."
        ),
    },
    "neutral_dialogue_01": {
        "id": "neutral_dialogue_01",
        "displayName": "Neutral Dialogue 01",
        "genderPresentation": "neutral",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral"],
        "backend": "parler",
        "licenseNotes": "Parler TTS Apache 2.0",
        "parlerDescription": (
            "A speaker with a clear, neutral voice delivers dialogue "
            "in a quiet studio environment. "
            "The recording has no background noise."
        ),
    },
}
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_tts.py::test_voice_registry_has_parler_descriptions -v
```

Expected: PASS

**Step 5: Run full test suite to check nothing broke**

```bash
.venv/bin/pytest -v
```

Expected: all existing tests pass.

**Step 6: Commit**

```bash
git add audiobook_worker/script_builder.py tests/test_tts.py
git commit -m "feat: add parlerDescription to voice registry entries"
```

---

### Task 3: Implement ParlerTTSBackend

**Files:**
- Modify: `audiobook_worker/tts.py`

**Step 1: Write the failing test**

In `tests/test_tts.py`, add:

```python
import os
from unittest.mock import MagicMock, patch


def test_parler_backend_synthesize_segment_produces_wav(tmp_path: Path):
    """ParlerTTSBackend.synthesize_segment writes a WAV and returns correct artifact."""
    import numpy as np

    fake_audio = np.zeros(24000, dtype=np.float32)  # 1 second at 24kHz

    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24000
    mock_model.generate.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: fake_audio.reshape(1, -1))
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    with patch("audiobook_worker.tts.ParlerTTSForConditionalGeneration") as mock_cls, \
         patch("audiobook_worker.tts.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from audiobook_worker.tts import ParlerTTSBackend
        backend = ParlerTTSBackend()

        segment = {
            "id": "seg_0001",
            "text": "It was a dark and stormy night.",
            "voiceId": "narrator_default",
            "emotion": "neutral",
            "intensity": 0.2,
            "pace": "normal",
        }
        artifact = backend.synthesize_segment(segment, tmp_path)

    assert artifact.kind == "segment_audio"
    assert artifact.path.suffix == ".wav"
    assert artifact.path.exists()
    assert artifact.duration_seconds > 0


def test_parler_backend_builds_description_with_emotion(tmp_path: Path):
    """Emotion modifiers are appended to the base voice description."""
    import numpy as np

    fake_audio = np.zeros(24000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24000
    mock_model.generate.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: fake_audio.reshape(1, -1))
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    with patch("audiobook_worker.tts.ParlerTTSForConditionalGeneration") as mock_cls, \
         patch("audiobook_worker.tts.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from audiobook_worker.tts import ParlerTTSBackend
        backend = ParlerTTSBackend()

        captured_descriptions = []
        original_call = mock_tokenizer.__call__

        segment = {
            "id": "seg_0002",
            "text": "Get out of my house!",
            "voiceId": "male_adult_01",
            "emotion": "angry",
            "intensity": 0.7,
            "pace": "fast",
        }
        backend.synthesize_segment(segment, tmp_path)

    # The tokenizer is called first with the description, second with the prompt
    first_call_args = mock_tokenizer.call_args_list[0][0][0]
    assert "angry" in first_call_args.lower() or "forceful" in first_call_args.lower()
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_tts.py::test_parler_backend_synthesize_segment_produces_wav tests/test_tts.py::test_parler_backend_builds_description_with_emotion -v
```

Expected: FAIL — `ImportError: cannot import name 'ParlerTTSBackend'`

**Step 3: Implement ParlerTTSBackend**

Replace `audiobook_worker/tts.py` entirely with:

```python
from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from audiobook_worker.script_builder import VOICE_REGISTRY

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------

_EMOTION_MODIFIERS: dict[str, str] = {
    "angry": "The speaker sounds angry and forceful, with sharp emphasis on stressed words.",
    "afraid": "The speaker sounds fearful and tense, with a slightly trembling, hushed delivery.",
    "sad": "The speaker sounds sorrowful and subdued, with a slow, quiet, measured pace.",
    "happy": "The speaker sounds warm and cheerful, with a light, upbeat cadence.",
    "excited": "The speaker sounds enthusiastic and energetic, speaking at a brisk, lively pace.",
    "tense": "The speaker sounds tense and guarded, with clipped, deliberate phrasing.",
    "neutral": "The speaker delivers the text clearly and evenly, without strong emotional colour.",
}

_PACE_MODIFIERS: dict[str, str] = {
    "slow": "The pace is slow and unhurried.",
    "normal": "",
    "fast": "The pace is quick and urgent.",
}

_DEFAULT_MODEL_ID = "parler-tts/parler-tts-mini-v1"


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
# Parler TTS backend
# ---------------------------------------------------------------------------

class ParlerTTSBackend:
    backend_id = "parler"

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def synthesize_segment(self, segment: dict, output_directory: Path | str) -> AudioArtifact:
        import soundfile as sf

        self._ensure_model()

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / f"{segment['id']}.wav"

        description = self._build_description(segment)
        text = segment["text"]

        audio_array = self._generate(description, text)

        sf.write(str(output_path), audio_array, self._model.config.sampling_rate)

        duration = len(audio_array) / self._model.config.sampling_rate
        return AudioArtifact(
            kind="segment_audio",
            path=output_path,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self._model_id
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)

    def _build_description(self, segment: dict) -> str:
        voice_id = segment.get("voiceId", "narrator_default")
        voice_entry = VOICE_REGISTRY.get(voice_id, VOICE_REGISTRY["narrator_default"])
        base = voice_entry.get("parlerDescription", "A clear speaker.")

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
            )

        return generation.cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def voice_registry() -> dict[str, dict]:
    return VOICE_REGISTRY.copy()


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
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_tts.py -v
```

Expected: all 4 tests pass (2 old + 2 new).

**Step 5: Run full suite**

```bash
.venv/bin/pytest -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add audiobook_worker/tts.py tests/test_tts.py
git commit -m "feat: implement ParlerTTSBackend with MPS support and emotion-aware descriptions"
```

---

### Task 4: Wire ParlerTTSBackend into the CLI

**Files:**
- Modify: `audiobook_worker/cli.py:11,121`

**Step 1: Write the failing test**

In `tests/test_cli.py`, find the `synthesize_segment_audio` test and add a variant:

```python
def test_synthesize_segment_audio_uses_parler_backend(tmp_path: Path):
    """CLI synthesize_segment_audio command selects ParlerTTSBackend when backend=parler."""
    import json
    from unittest.mock import patch, MagicMock
    import numpy as np
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0001",
                "text": "In the beginning.",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "intensity": 0.2,
                "pace": "normal",
            }
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))

    request = {
        "scriptPath": str(script_path),
        "segmentId": "seg_0001",
        "outputDirectory": str(tmp_path / "audio"),
        "backend": "parler",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request))
    output_path = tmp_path / "output.json"

    fake_audio = np.zeros(24000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24000
    mock_model.generate.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: fake_audio.reshape(1, -1))
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    with patch("audiobook_worker.tts.ParlerTTSForConditionalGeneration") as mock_cls, \
         patch("audiobook_worker.tts.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        exit_code = main(["synthesize_segment_audio", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_cli.py::test_synthesize_segment_audio_uses_parler_backend -v
```

Expected: FAIL — CLI always uses `MockTTSBackend`, never Parler.

**Step 3: Update `_synthesize_segment_audio` in cli.py**

Replace the import at the top:
```python
from audiobook_worker.tts import MockTTSBackend, ParlerTTSBackend
```

Replace the `_synthesize_segment_audio` function:
```python
def _synthesize_segment_audio(request: dict[str, Any]) -> dict[str, Any]:
    script = json.loads(Path(request["scriptPath"]).read_text(encoding="utf-8"))
    segment = next(
        item for item in script["segments"] if item["id"] == request["segmentId"]
    )
    backend_name = request.get("backend", "mock")
    if backend_name == "parler":
        backend = ParlerTTSBackend()
    else:
        backend = MockTTSBackend()
    artifact = backend.synthesize_segment(segment, Path(request["outputDirectory"]))
    return _response(
        "succeeded",
        artifacts=[
            {
                "kind": artifact.kind,
                "path": str(artifact.path),
                "metadata": {"durationSeconds": artifact.duration_seconds},
            }
        ],
    )
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

Expected: all CLI tests pass.

**Step 5: Commit**

```bash
git add audiobook_worker/cli.py tests/test_cli.py
git commit -m "feat: wire ParlerTTSBackend into CLI via backend request field"
```

---

### Task 5: Download a test book and run end-to-end

**Goal:** Generate real speech from a real book chapter using Parler TTS on MPS.

**Step 1: Download a public domain EPUB**

Download *Pride and Prejudice* from Project Gutenberg (plain text, no DRM):

```bash
cd .worktrees/mvp-implementation/workers/python
curl -o /tmp/pride_and_prejudice.txt \
  "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
```

Expected: ~700KB plain text file.

**Step 2: Extract chapter 1 text**

```bash
.venv/bin/python - <<'EOF'
text = open("/tmp/pride_and_prejudice.txt").read()
# Find Chapter 1 start and Chapter 2 start
start = text.find("Chapter 1")
if start == -1:
    start = text.find("CHAPTER 1")
end = text.find("Chapter 2", start + 1)
if end == -1:
    end = text.find("CHAPTER 2", start + 1)
chapter1 = text[start:end].strip()
open("/tmp/ch01.txt", "w").write(chapter1)
print(f"Extracted {len(chapter1)} chars, first 200:\n{chapter1[:200]}")
EOF
```

Expected: prints ~2000-4000 chars of chapter 1 text.

**Step 3: Run analyze_chapter to build the script IR**

```bash
mkdir -p /tmp/audiobook_test/scripts

cat > /tmp/analyze_input.json <<'EOF'
{
  "bookId": "pride_prejudice",
  "chapterId": "ch01",
  "title": "Chapter 1",
  "chapterTextPath": "/tmp/ch01.txt",
  "outputDirectory": "/tmp/audiobook_test/scripts",
  "language": "en"
}
EOF

.venv/bin/audiobook-worker analyze_chapter \
  /tmp/analyze_input.json \
  /tmp/analyze_output.json

cat /tmp/analyze_output.json
```

Expected: `"status": "succeeded"` with `chapter_script` artifact path.

**Step 4: Inspect the script to pick a test segment**

```bash
.venv/bin/python - <<'EOF'
import json
script = json.load(open("/tmp/audiobook_test/scripts/ch01.json"))
for seg in script["segments"][:5]:
    print(f"[{seg['id']}] type={seg['type']} emotion={seg['emotion']} voice={seg['voiceId']}")
    print(f"  text: {seg['text'][:80]}")
    print()
EOF
```

Expected: list of 5 segments showing narrator and dialogue types.

**Step 5: Synthesize the first narration segment with Parler TTS**

Pick `seg_0001` (adjust id from step 4 output if needed).

```bash
cat > /tmp/synth_input.json <<'EOF'
{
  "scriptPath": "/tmp/audiobook_test/scripts/ch01.json",
  "segmentId": "seg_0001",
  "outputDirectory": "/tmp/audiobook_test/audio",
  "backend": "parler"
}
EOF

.venv/bin/audiobook-worker synthesize_segment_audio \
  /tmp/synth_input.json \
  /tmp/synth_output.json

cat /tmp/synth_output.json
```

Expected: `"status": "succeeded"` with a `.wav` path. First run downloads the model (~400MB).

**Step 6: Play back the generated audio**

```bash
afplay /tmp/audiobook_test/audio/seg_0001.wav
```

Expected: hear the narration spoken by a clear male voice.

**Step 7: Synthesize a dialogue segment (if one exists)**

From step 4, pick a `type: dialogue` segment id. Replace `seg_0001` with that id:

```bash
cat > /tmp/synth_dialogue_input.json <<'EOF'
{
  "scriptPath": "/tmp/audiobook_test/scripts/ch01.json",
  "segmentId": "seg_0003",
  "outputDirectory": "/tmp/audiobook_test/audio",
  "backend": "parler"
}
EOF

.venv/bin/audiobook-worker synthesize_segment_audio \
  /tmp/synth_dialogue_input.json \
  /tmp/synth_dialogue_output.json

afplay /tmp/audiobook_test/audio/seg_0003.wav
```

Expected: different voice from narrator.

**Step 8: Assemble all synthesized segments into one chapter WAV**

First synthesize all segments (first 10 to keep it fast):

```bash
.venv/bin/python - <<'EOF'
import json, subprocess, pathlib

script = json.load(open("/tmp/audiobook_test/scripts/ch01.json"))
out_dir = pathlib.Path("/tmp/audiobook_test/audio")
out_dir.mkdir(parents=True, exist_ok=True)

for seg in script["segments"][:10]:
    inp = {
        "scriptPath": "/tmp/audiobook_test/scripts/ch01.json",
        "segmentId": seg["id"],
        "outputDirectory": str(out_dir),
        "backend": "parler",
    }
    inp_path = f"/tmp/synth_{seg['id']}.json"
    out_path = f"/tmp/synth_{seg['id']}_out.json"
    json.dump(inp, open(inp_path, "w"))
    result = subprocess.run(
        [".venv/bin/audiobook-worker", "synthesize_segment_audio", inp_path, out_path],
        capture_output=True, text=True
    )
    status = json.load(open(out_path)).get("status")
    print(f"{seg['id']}: {status}")
EOF
```

Then assemble:

```bash
cat > /tmp/assemble_input.json <<'EOF'
{
  "segmentAudioDirectory": "/tmp/audiobook_test/audio",
  "outputPath": "/tmp/audiobook_test/ch01_chapter.wav"
}
EOF

.venv/bin/audiobook-worker assemble_chapter_audio \
  /tmp/assemble_input.json \
  /tmp/assemble_output.json

cat /tmp/assemble_output.json
afplay /tmp/audiobook_test/ch01_chapter.wav
```

Expected: hear a ~1-2 minute assembled chapter with different voices for narrator vs dialogue.

---

### Task 6: Final cleanup and commit

**Step 1: Run full test suite one last time**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest -v
```

Expected: all tests pass.

**Step 2: Commit any remaining changes**

```bash
git add -p
git commit -m "test: end-to-end Parler TTS verification with Pride and Prejudice ch01"
```
