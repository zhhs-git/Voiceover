# Audiobook Generator Design

## Status

Draft. This document defines the initial product and technical direction for a desktop/local audiobook generator.

## Summary

The application converts user-provided PDF or EPUB books into chapter-based audiobooks. It runs primarily on the user's machine, with local OCR and local TTS as the default operating mode. TypeScript should own the desktop app, UI, orchestration, job state, and user-facing workflows. Python is acceptable for OCR, NLP, TTS, and AI model integration where the ecosystem is stronger.

The core design choice is to avoid sending raw book text directly to TTS. The app first converts a book into a structured audiobook script that separates narration from dialogue, assigns speakers and voices, tags emotion, and records confidence. Audio generation then consumes this script per chapter or per segment.

## Goals

- Generate audiobooks from PDF and EPUB files.
- Prefer local processing for privacy, cost control, and offline use.
- Support chapter-based generation, retries, and regeneration.
- Infer dialogue speakers, character gender, and emotion with minimal user intervention.
- Assign suitable voices automatically, such as female voices for female characters when confidence is high.
- Provide optional review for low-confidence issues without requiring line-by-line editing.
- Keep generated assets organized by book and chapter.
- Leave room for multiple OCR, LLM, and TTS backends.

## Non-Goals

- The first version will not guarantee perfect speaker attribution.
- The first version will not solve copyright law automatically.
- The first version will not require a hosted backend.
- The first version will not attempt full professional audiobook mastering.
- The first version will not support every possible book layout, DRM format, or scanned document quality.

## Product Shape

The initial product should be a local desktop app. The user opens the app, imports a PDF or EPUB, reviews a short analysis summary, optionally listens to a preview, then starts audiobook generation.

The primary workflow should be:

1. Import book.
2. Extract or OCR text.
3. Detect chapters.
4. Build character bible.
5. Build audiobook script.
6. Assign voices and emotions.
7. Generate a short preview.
8. Generate chapter audio.
9. Export chapter files and metadata.

The app should not force the user to correct every uncertain line. It should proceed with conservative fallbacks and surface optional review only when confidence is low enough to affect output quality.

## Recommended Technology Stack

### Desktop Shell

Use TypeScript with Tauri or Electron.

Tauri is the recommended default because it produces a smaller desktop app and pairs well with a local sidecar process. Electron remains a fallback if browser or media APIs become easier there.

Recommended:

- Tauri for desktop packaging.
- TypeScript for app orchestration and UI.
- React or Svelte for the renderer UI.
- SQLite for local job state, metadata, and script storage.

### UI and Orchestration

Use TypeScript for:

- Import workflow.
- Book/job dashboard.
- Chapter progress.
- Character and voice summary.
- Optional review screens.
- Calling Python workers.
- Managing job status, retries, and cancellation.
- Export and file organization.

### AI, OCR, and TTS Workers

Use Python for:

- OCR integration.
- EPUB/PDF text extraction fallback logic.
- Dialogue and speaker analysis if using local NLP/LLM tooling.
- Local TTS model integration.
- Audio segment generation and stitching when Python libraries are strongest.

The TypeScript app should treat Python as a worker runtime with stable command/API boundaries rather than mixing Python deeply into UI logic.

### Candidate Libraries

Text extraction:

- EPUB: `ebooklib`, `beautifulsoup4`, or a TypeScript EPUB parser if adequate.
- PDF selectable text: `pymupdf` or `pdfplumber`.
- Scanned PDF OCR: OCRmyPDF, Tesseract, PaddleOCR, or docTR depending on quality/performance tradeoffs.

Dialogue and analysis:

- Local LLM through Ollama, llama.cpp, LM Studio, or vLLM when available.
- Hosted LLM can be optional later, but not required for the local-first design.
- Rule-based dialogue parsing should be used before LLM calls to reduce cost and improve consistency.

TTS:

- Coqui XTTS-style models, Piper, StyleTTS2-family models, or other local engines depending on voice quality and hardware.
- The app should support a pluggable TTS backend interface because model quality and licensing change often.

Audio processing:

- `ffmpeg` for concatenation, transcoding, silence trimming, and MP3/M4B export.
- `pydub` or direct ffmpeg calls from Python for implementation convenience.

## Architecture

The app should be split into four layers:

1. Desktop UI
2. TypeScript orchestration service
3. Python processing workers
4. Local model and file storage

The UI should never call model code directly. It should call the TypeScript orchestration layer, which schedules work and invokes workers.

The Python worker boundary should use either:

- CLI commands with JSON input/output for the MVP.
- A local HTTP/gRPC worker service if jobs become long-lived or concurrent.

CLI JSON is simpler for the first implementation. A service can be introduced later if startup time, streaming progress, or cancellation becomes painful.

## Data Flow

### Import

The app creates a book record and stores the original file path or an imported copy under the app data directory.

### Extraction

The extraction worker determines whether the file is EPUB, selectable PDF, or scanned PDF.

Output:

- raw extracted text
- detected pages or EPUB spine items
- extraction warnings
- OCR confidence when applicable

### Chapter Detection

The chapter worker segments the book into chapters using EPUB structure, table of contents, headings, or text heuristics.

Output:

- chapter id
- title
- source range
- cleaned text
- confidence

### Character Bible

The analysis worker scans the book or a representative subset and creates character profiles.

Each character profile should include:

- canonical name
- aliases
- likely gender
- likely age class
- speaking style notes
- voice assignment
- confidence

### Audiobook Script

Each chapter is transformed into a script containing ordered segments.

Segment fields:

- id
- chapter id
- type: narration, dialogue, heading, silence, sound_cue
- text
- speaker id
- voice id
- emotion
- intensity
- pace
- confidence
- source location
- warnings

### Audio Generation

TTS consumes the script by segment. Generated segment audio is cached. Chapter audio is assembled from segment files so failed work can resume and individual segments can be regenerated.

Output:

- segment audio files
- chapter audio files
- export files
- generation logs

## Script Intermediate Representation

The script IR is the central contract between analysis and audio generation. It should be stored as structured JSON in SQLite or as JSON files referenced by SQLite.

Example:

```json
{
  "bookId": "book_123",
  "chapterId": "chapter_003",
  "segments": [
    {
      "id": "seg_0001",
      "type": "narration",
      "speakerId": "narrator",
      "voiceId": "narrator_default",
      "text": "She opened the door slowly.",
      "emotion": "tense",
      "intensity": 0.4,
      "pace": "slow",
      "confidence": 0.93
    },
    {
      "id": "seg_0002",
      "type": "dialogue",
      "speakerId": "elizabeth",
      "voiceId": "female_adult_01",
      "text": "Who's there?",
      "emotion": "afraid",
      "intensity": 0.7,
      "pace": "normal",
      "confidence": 0.82
    }
  ]
}
```

## Dialogue and Emotion Awareness

The app should combine deterministic parsing with LLM inference.

Deterministic parsing should handle:

- quoted text extraction
- nearby speech tags
- paragraph boundaries
- obvious speaker names
- punctuation and casing cues

LLM analysis should handle:

- ambiguous speaker attribution
- alias merging
- gender inference from context
- emotion classification
- speaking style summaries

The system should process at chapter or scene scale rather than whole-book prompts when possible. Whole-book analysis can build the character bible, but final segment decisions should be tied to local context to reduce hallucination.

## Voice Assignment

Voices should be assigned from a local voice registry. Each voice should declare:

- id
- display name
- gender presentation
- age class
- language support
- style/emotion capability
- backend
- license notes

The assignment algorithm should:

1. Reserve a narrator voice.
2. Assign major characters distinct voices.
3. Match likely gender and age when confidence is high.
4. Avoid using the same voice for two major characters in the same scene.
5. Use neutral fallback voices for low-confidence characters.

## Language Support

The first version should distinguish between source language and output language.

Modes:

- Same-language narration: read the book in its original language.
- Translated narration: translate script text before TTS.

Translation should happen after chapter/script segmentation, not on raw full-book text. This preserves speakers, emotions, and chapter structure.

Language support depends on local TTS voices. The app should only offer output languages with installed voices or clearly mark that setup is required.

## License and Rights Flow

The app should perform best-effort license checks but must not claim to provide legal certainty.

Checks:

- EPUB metadata rights fields.
- PDF metadata when present.
- visible Creative Commons or public-domain notices.
- Project Gutenberg indicators.
- publication year and author metadata where available.
- DRM detection.

The app should classify uploads as:

- allowed
- restricted
- unknown
- blocked

For unknown or restricted files, the app should ask the user to attest that they have the right to convert the work for their intended use.

## Error Handling

The app should treat the pipeline as resumable.

Each stage should write explicit status:

- pending
- running
- succeeded
- failed
- skipped
- needs_review

Failures should be scoped to the smallest practical unit:

- extraction failure blocks the book.
- chapter detection failure blocks script generation.
- TTS segment failure should not discard completed segments.
- chapter assembly failure should preserve segment audio.

The app should preserve logs for each stage and show concise user-facing errors with a path to detailed logs.

## User Intervention Strategy

The default should be automatic generation. Review is optional unless the app cannot proceed.

The app should flag:

- poor OCR quality
- missing chapters
- many unknown speakers
- low-confidence major character gender
- unavailable voice for target language
- license status blocked or unknown

The app should allow global corrections:

- merge aliases
- set character gender
- assign voice
- mark a speaker as narrator-only
- regenerate affected chapters

## Storage Layout

Use an app data directory with a predictable structure:

```text
books/
  <book-id>/
    source/
    extracted/
    scripts/
    audio/
      segments/
      chapters/
    exports/
    logs/
```

Use SQLite for:

- books
- chapters
- characters
- voices
- jobs
- stage status
- warnings
- license attestations

Large artifacts should stay on disk and be referenced from SQLite.

## MVP Scope

The MVP should include:

- desktop/local app shell
- PDF and EPUB import
- EPUB extraction
- selectable-text PDF extraction
- optional OCR path for scanned PDFs if OCR tooling is available
- chapter detection
- character bible generation
- dialogue/narration segmentation
- emotion tagging
- automatic voice assignment
- per-chapter TTS generation
- MP3 chapter export
- basic progress UI
- optional character voice review

The MVP can defer:

- M4B export
- advanced mastering
- cloud sync
- mobile apps
- collaborative review
- custom voice cloning
- hosted generation

## Testing Strategy

Use fixture books with known structure:

- clean EPUB
- selectable-text PDF
- scanned PDF sample
- dialogue-heavy chapter
- multi-character scene
- non-English sample

Test levels:

- unit tests for parsers and schema validation
- integration tests for stage inputs and outputs
- golden fixtures for script IR shape
- smoke tests for local TTS generation
- manual audio review for voice and emotion quality

Because LLM outputs are nondeterministic, tests should validate schema, confidence behavior, and basic invariants rather than exact prose.

## Security and Privacy

The default assumption is that book contents remain local.

If hosted APIs are added later, the app must clearly show:

- what text is sent
- to which provider
- for which purpose
- whether the provider may retain data

Local files should not be uploaded silently.

## Open Questions

- Which TTS backend gives the best local quality for the target hardware?
- Should the first UI use Tauri plus React, or Electron plus React?
- Should the Python worker start per job or run as a persistent local service?
- Which languages are required for the first release?
- What minimum hardware should be supported?
