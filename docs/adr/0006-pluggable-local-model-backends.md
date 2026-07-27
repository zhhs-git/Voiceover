# ADR 0006: Pluggable Local Model Backends

## Status

Accepted.

## Context

Local OCR, LLM, and TTS tooling changes quickly. Voice quality, language support, hardware requirements, and model licenses vary significantly. Locking the app to one model would make the system brittle.

## Decision

Use pluggable backend interfaces for OCR, LLM analysis, and TTS.

Each backend should declare:

- backend id
- supported task type
- supported languages
- hardware requirements
- license notes
- configuration schema
- health check command

The app should select sensible defaults but keep the pipeline independent from any single model provider.

## Consequences

Benefits:

- Allows improving quality without redesigning the app.
- Supports different hardware levels.
- Makes language expansion easier.
- Enables optional hosted backends later without changing the script IR.

Costs:

- Requires stable backend contracts.
- Adds configuration complexity.
- Testing must cover backend mocks and at least one real backend.

## Alternatives Considered

Single blessed backend:

- Easier MVP.
- Risky because model quality and licensing may change.

Hosted-only backend:

- Higher quality may be easier.
- Conflicts with local-first privacy and cost goals.
