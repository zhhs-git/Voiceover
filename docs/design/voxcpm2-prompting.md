# VoxCPM2 Prompting Contract

This document describes the local VoxCPM2 audiobook path. It is an adapter
contract, not a second role-design workflow.

## One Canonical Voice Design

The existing `voice_design` analysis stage is the only author of the durable
role field:

```json
{
  "id": "character_id",
  "voiceDesign": "A stable natural-language description of the role's voice"
}
```

The same field is stored in the character roster, script, review data, and
TTS segment inputs. Existing non-empty values are preserved verbatim. No
VoxCPM2-specific field is shown in the UI, and no additional LLM call
translates or rewrites the design.

The design describes durable audible identity only: speaker type and age
impression, timbre, pitch/range, resonance, diction, breath, cadence, and
stable differentiation from other roles. It excludes temporary emotion, pace,
scene direction, music, effects, and post-processing.

Narrator identity is resolved from the selected narrator voice ID. A segment's
`voiceDesign` cannot replace that selected narrator identity.

## Profile Reference

The main worker deterministically projects the canonical design into a compact
`profileControl`. Presentation labels such as `Role:` and `角色：` are removed,
whitespace is normalized, and the result is bounded to 180 characters. For a
Chinese script, durable Chinese descriptors are retained. For a non-Chinese
script, a small deterministic English fallback retains recognizable gender,
age, timbre, resonance, and diction anchors; the stored `voiceDesign` is not
changed.

The isolated runner sends exactly this effective profile text to VoxCPM2:

```text
(<profileControl>)<referenceText>
```

The reference sentence is fixed and neutral:

```text
zh: 清晨的风穿过窗边，屋里很安静。
en: The morning light falls softly across the quiet room.
```

The profile request also carries `voiceDesign`, `profileControl`, `language`,
and `promptFormatVersion` for sidecar observability. Only `profileControl` is
used as the model control text; `voiceDesign` is metadata, not a second
prompt.

## Reference-Profile Loudness

VoxCPM2 normalizes only the generated reference profile before it enters the
voice-profile cache. The runner writes model output to a temporary candidate,
uses ffmpeg's `loudnorm` filter, validates the resulting PCM S16 WAV, and only
then atomically replaces the accepted reference file. The fixed contract is:

```text
integrated loudness: -20 LUFS
true peak ceiling:   -3 dBFS
loudness range:      7 LU
profileLoudness:     version 1
```

The profile payload and sidecar both persist the versioned mapping:

```json
{
  "profileLoudness": {
    "version": 1,
    "integratedLufs": -20.0,
    "truePeakDb": -3.0,
    "loudnessRange": 7.0
  }
}
```

This is deliberately a base-level correction for role identity. It does not
normalize individual cloned segments, the assembled chapter, or the final mix:
whispers, pauses, emotion, and pace remain part of each segment's performance.
If ffmpeg fails, only the candidate is removed; an earlier accepted profile and
sidecar remain untouched.

## Segment Delivery

Every segment reuses the profile WAV for stable identity. Its dynamic control
contains only the current analysis products:

```text
(<localized emotion>, <localized pace>[, <bounded voiceDirection>])<source text>
```

`voiceDirection` is normalized and bounded to 120 characters. Scene context,
the canonical voice design, and identity-locking prose are never copied into
this control. Chinese segments use concise Chinese emotion and pace phrases;
English and other segments use English fallback phrases. The complete script
language is attached to each decorated TTS segment before this composition.

Examples:

```text
(谨慎克制，语速偏快但字音清楚，句首短暂停顿)现在必须离开。
(guarded and quietly pressured, quick but clearly articulated)The door opened.
```

## Runner and Cache Contract

`promptFormatVersion = 2` remains part of every local runner request, profile
sidecar, and VoxCPM2 segment cache signature. `profileLoudness = v1` is a
separate local-only cache contract: it is included in local profile and segment
signatures, and is required in local profile sidecars. Old unnormalized local
profiles and dependent segment WAVs therefore miss on their next original
chapter regeneration. MiMo signature bytes and request structure are unchanged,
and already completed chapter/final audio is never rewritten in place.

One chapter still makes one isolated runner request. That process loads
VoxCPM2 once and synthesizes uncached segments serially in source order. The
model parameters remain `cfg_value=2.0` and `inference_timesteps=10`.

Independent chapters are admitted through the web server's shared `voxcpm`
resource, but the active capacity is one whole-chapter runner. Batch and direct
TTS requests use the same gate. `AUDIOBOOK_VOXCPM_WORKER_CONCURRENCY` is kept
for compatibility and is clamped to one for every value. Two-way runs
contended for the shared MPS device and produced malformed WAVs without a
stable wall-clock gain. Segment-level parallelism is intentionally unsupported.

## Why Controllable Cloning

The normal audiobook path uses controllable cloning: the reference WAV carries
stable timbre and each segment can independently express emotion, pace, and
delivery. Ultimate cloning would reproduce a reference performance more
literally, but it conflicts with the installed model interface when a style
control is supplied with the exact reference transcript. Using it for every
segment would also carry one performance into unrelated text. Keeping the
reference-only profile and compact per-segment control preserves the intended
separation.
