# ADR 0003: Script Intermediate Representation

## Status

Accepted.

## Context

Plain TTS over extracted book text cannot reliably support dialogue-aware and emotion-aware audiobook generation. The app must know which text is narration, which text is dialogue, who is speaking, what voice to use, and how the line should be performed.

Audio generation also needs to be resumable per chapter and per segment.

## Decision

Convert each book into a structured audiobook script before TTS generation.

The script is the contract between analysis and audio generation. It contains chapters, segments, speaker IDs, voice IDs, emotions, pacing hints, confidence values, warnings, and source locations.

TTS generation consumes this script rather than raw book text.

## Consequences

Benefits:

- Enables dialogue-aware voice selection.
- Enables emotion and pacing control.
- Makes audio generation resumable.
- Allows targeted regeneration after corrections.
- Provides a reviewable artifact for debugging and user edits.

Costs:

- Adds a preprocessing stage before audio can be generated.
- Requires schema versioning.
- Requires handling uncertainty and partial correctness.

## Alternatives Considered

Direct chapter-to-TTS:

- Faster to build.
- Cannot support reliable speaker voices or emotion direction.

Prompt-only TTS instructions:

- Simple if the TTS backend supports rich prompts.
- Hard to validate, review, resume, or correct.

Manual script editor:

- High quality.
- Too much user intervention for the intended product.
