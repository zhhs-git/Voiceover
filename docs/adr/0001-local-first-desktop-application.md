# ADR 0001: Local-First Desktop Application

## Status

Accepted.

## Context

The app generates audiobooks from user-provided PDF and EPUB files. Book content can be private, copyrighted, or otherwise sensitive. Full-book processing can also be expensive if every OCR, LLM, or TTS operation depends on hosted services.

The preferred product shape is a desktop/local app. The user prefers TypeScript for application development, while Python is acceptable for TTS and AI work.

## Decision

Build the first version as a local-first desktop application.

Use TypeScript for the desktop app, UI, orchestration, job state, and integration boundaries. Use Python for OCR, AI/NLP analysis, local model integration, TTS generation, and audio processing where Python libraries are the practical choice.

The recommended desktop shell is Tauri unless implementation constraints make Electron more practical.

## Consequences

Benefits:

- User book content stays local by default.
- Marginal generation cost is mostly electricity and hardware time.
- The app can work offline after models and dependencies are installed.
- Local files and model caches can be reused across runs.

Costs:

- Installation and dependency management are harder than a hosted web app.
- Local model quality depends on the user's hardware.
- Cross-platform packaging requires more care.
- GPU acceleration and model setup may vary across machines.

## Alternatives Considered

Hosted web app:

- Easier to update and support.
- Better control over model hardware.
- Worse for privacy and operating cost.
- Requires upload/storage policy and stronger compliance controls.

CLI-only app:

- Fastest to build.
- Useful for power users.
- Poor fit for review, progress, and voice assignment workflows.

Local web app:

- Easier than full desktop packaging.
- Still requires a local server and browser workflow.
- Less polished for a consumer-style tool.
