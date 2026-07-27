# ADR 0002: TypeScript Orchestration With Python Workers

## Status

Accepted.

## Context

The app needs a desktop UI, persistent local state, background jobs, OCR, text extraction, dialogue analysis, TTS generation, and audio stitching.

TypeScript is preferred for app development. Python has stronger libraries for OCR, machine learning, local TTS, NLP, and audio processing.

## Decision

Use TypeScript as the main application language and Python as a worker language.

TypeScript owns:

- desktop application shell
- renderer UI
- job orchestration
- SQLite state
- progress reporting
- user settings
- worker invocation
- export workflow

Python owns:

- OCR
- PDF/EPUB extraction where Python libraries are better
- LLM/NLP analysis
- local TTS backend integration
- audio segment generation
- audio stitching helpers

The worker boundary should use JSON schemas. The first version may invoke workers as CLI commands. A persistent local service can replace CLI workers later if needed.

## Consequences

Benefits:

- Keeps UI and product code in TypeScript.
- Uses Python where the AI/audio ecosystem is strongest.
- Creates a clean boundary between orchestration and heavy processing.
- Allows replacing OCR/TTS/LLM backends without rewriting the UI.

Costs:

- Packaging must include both Node/TypeScript and Python runtime concerns.
- Schema compatibility must be maintained.
- Worker logs and errors must be normalized for the UI.

## Alternatives Considered

All TypeScript:

- Simpler packaging and one language.
- Weaker access to local AI, OCR, and TTS tooling.

All Python:

- Strong model ecosystem.
- Less ideal for a polished desktop app and TypeScript-preferred development.

Rust-heavy Tauri backend:

- Efficient and clean for native integration.
- Adds complexity without replacing the need for Python model tooling.
