# Audiobook Generator MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first local desktop audiobook generator that imports PDF/EPUB books, creates a dialogue-aware script, assigns voices, and generates chapter audio locally.

**Architecture:** Use a TypeScript desktop app for UI, orchestration, local state, and job control. Use Python workers for extraction, OCR, analysis, TTS, and audio processing behind JSON contracts. Store job state and metadata in SQLite, and store large artifacts on disk under a per-book directory.

**Tech Stack:** Tauri, TypeScript, React or Svelte, SQLite, Python, PyMuPDF/pdfplumber, ebooklib, optional OCRmyPDF/Tesseract/PaddleOCR, local LLM backend, local TTS backend, ffmpeg.

---

## Phase 0: Repository Setup

### Task 1: Initialize Project Structure

**Files:**
- Create: `apps/desktop/`
- Create: `workers/python/`
- Create: `packages/shared/`
- Create: `fixtures/books/`
- Create: `docs/design/`
- Create: `docs/adr/`
- Create: `docs/plans/`

**Step 1: Initialize git**

Run: `git init`

Expected: repository initialized.

**Step 2: Create base directories**

Run: `mkdir -p apps/desktop workers/python packages/shared fixtures/books docs/design docs/adr docs/plans`

Expected: directories exist.

**Step 3: Add root README**

Create `README.md`:

```markdown
# Audiobook Generator

Local-first desktop app for generating chapter-based audiobooks from PDF and EPUB files.
```

**Step 4: Commit**

```bash
git add README.md apps workers packages fixtures docs
git commit -m "chore: initialize audiobook generator repository"
```

## Phase 1: Shared Contracts

### Task 2: Define Script IR Schema

**Files:**
- Create: `packages/shared/src/script-ir.ts`
- Create: `packages/shared/src/script-ir.test.ts`
- Create: `packages/shared/package.json`

**Step 1: Write schema tests**

Test that a valid chapter script accepts narration and dialogue segments with speaker, voice, emotion, and confidence fields.

**Step 2: Implement TypeScript types and validation**

Define:

- `BookScript`
- `ChapterScript`
- `ScriptSegment`
- `CharacterProfile`
- `VoiceProfile`
- `Emotion`
- `SegmentType`

Use a runtime schema library such as `zod`.

**Step 3: Run tests**

Run: `npm test --workspace packages/shared`

Expected: schema tests pass.

**Step 4: Commit**

```bash
git add packages/shared
git commit -m "feat: define audiobook script IR schema"
```

### Task 3: Define Worker JSON Protocols

**Files:**
- Create: `packages/shared/src/workers.ts`
- Create: `packages/shared/src/workers.test.ts`

**Step 1: Write tests**

Cover request/response schemas for:

- extract book text
- detect chapters
- analyze chapter script
- synthesize segment audio
- assemble chapter audio

**Step 2: Implement schemas**

Each worker response must include:

- `status`
- `warnings`
- `artifacts`
- `error` when failed

**Step 3: Run tests**

Run: `npm test --workspace packages/shared`

Expected: worker protocol tests pass.

**Step 4: Commit**

```bash
git add packages/shared
git commit -m "feat: define worker JSON protocols"
```

## Phase 2: Python Worker Foundation

### Task 4: Create Python Worker CLI

**Files:**
- Create: `workers/python/pyproject.toml`
- Create: `workers/python/audiobook_worker/__init__.py`
- Create: `workers/python/audiobook_worker/cli.py`
- Create: `workers/python/tests/test_cli.py`

**Step 1: Write CLI tests**

Test that `audiobook-worker --help` works and unknown commands return a structured error.

**Step 2: Implement CLI entrypoint**

Use `argparse` or `typer`. Accept JSON input path and JSON output path.

**Step 3: Run tests**

Run: `pytest workers/python/tests -v`

Expected: CLI tests pass.

**Step 4: Commit**

```bash
git add workers/python
git commit -m "feat: add python worker CLI foundation"
```

### Task 5: Implement EPUB and PDF Text Extraction

**Files:**
- Create: `workers/python/audiobook_worker/extract.py`
- Create: `workers/python/tests/test_extract.py`

**Step 1: Add fixtures**

Add one tiny EPUB fixture and one tiny selectable PDF fixture under `fixtures/books/`.

**Step 2: Write extraction tests**

Assert that extraction returns normalized text and source metadata.

**Step 3: Implement extraction**

Use `ebooklib` for EPUB and `pymupdf` or `pdfplumber` for selectable PDFs.

**Step 4: Run tests**

Run: `pytest workers/python/tests/test_extract.py -v`

Expected: extraction tests pass.

**Step 5: Commit**

```bash
git add workers/python fixtures/books
git commit -m "feat: extract text from epub and pdf"
```

### Task 6: Add OCR Detection Hook

**Files:**
- Modify: `workers/python/audiobook_worker/extract.py`
- Create: `workers/python/audiobook_worker/ocr.py`
- Create: `workers/python/tests/test_ocr_detection.py`

**Step 1: Write tests**

Test that a PDF with too little selectable text is marked as requiring OCR.

**Step 2: Implement OCR detection**

Add a detector that classifies PDFs as selectable text, scanned, or mixed.

**Step 3: Add placeholder OCR backend**

Return a clear `ocr_backend_not_configured` error until an OCR backend is installed.

**Step 4: Run tests**

Run: `pytest workers/python/tests/test_ocr_detection.py -v`

Expected: OCR detection tests pass.

**Step 5: Commit**

```bash
git add workers/python
git commit -m "feat: detect scanned PDFs for OCR"
```

## Phase 3: Analysis Pipeline

### Task 7: Implement Chapter Detection

**Files:**
- Create: `workers/python/audiobook_worker/chapters.py`
- Create: `workers/python/tests/test_chapters.py`

**Step 1: Write tests**

Use sample text with headings like `Chapter 1`, `CHAPTER II`, and EPUB title markers.

**Step 2: Implement heuristics**

Detect chapters from EPUB TOC when available, then heading heuristics as fallback.

**Step 3: Run tests**

Run: `pytest workers/python/tests/test_chapters.py -v`

Expected: chapter detection tests pass.

**Step 4: Commit**

```bash
git add workers/python
git commit -m "feat: detect chapters from extracted text"
```

### Task 8: Implement Rule-Based Dialogue Segmentation

**Files:**
- Create: `workers/python/audiobook_worker/dialogue.py`
- Create: `workers/python/tests/test_dialogue.py`

**Step 1: Write tests**

Cover narration, quoted dialogue, speech tags, and alternating dialogue.

**Step 2: Implement segmentation**

Split chapter text into narration and dialogue segments. Preserve source order and source offsets.

**Step 3: Run tests**

Run: `pytest workers/python/tests/test_dialogue.py -v`

Expected: segmentation tests pass.

**Step 4: Commit**

```bash
git add workers/python
git commit -m "feat: segment narration and dialogue"
```

### Task 9: Add LLM Analysis Adapter

**Files:**
- Create: `workers/python/audiobook_worker/llm.py`
- Create: `workers/python/tests/test_llm_adapter.py`

**Step 1: Write mock adapter tests**

Test that the adapter accepts chapter context and returns character, speaker, emotion, and confidence fields.

**Step 2: Implement mock backend**

Start with a deterministic mock backend so the rest of the app can be built without a real model.

**Step 3: Add real backend interface**

Define configuration for Ollama or another local LLM provider, but keep it disabled by default.

**Step 4: Run tests**

Run: `pytest workers/python/tests/test_llm_adapter.py -v`

Expected: adapter tests pass.

**Step 5: Commit**

```bash
git add workers/python
git commit -m "feat: add LLM analysis adapter"
```

### Task 10: Generate Audiobook Script

**Files:**
- Create: `workers/python/audiobook_worker/script_builder.py`
- Create: `workers/python/tests/test_script_builder.py`

**Step 1: Write tests**

Assert that a chapter becomes valid script JSON with narration, dialogue, speakers, emotions, voices, and confidence.

**Step 2: Implement script builder**

Combine chapter text, dialogue segmentation, character analysis, and voice assignment into script IR.

**Step 3: Validate output**

Validate against the shared schema or a mirrored Python schema.

**Step 4: Run tests**

Run: `pytest workers/python/tests/test_script_builder.py -v`

Expected: script builder tests pass.

**Step 5: Commit**

```bash
git add workers/python
git commit -m "feat: build dialogue-aware audiobook scripts"
```

## Phase 4: Local TTS and Audio

### Task 11: Create TTS Backend Interface

**Files:**
- Create: `workers/python/audiobook_worker/tts.py`
- Create: `workers/python/tests/test_tts.py`

**Step 1: Write tests**

Test that a mock backend generates a segment artifact path for a script segment.

**Step 2: Implement mock TTS backend**

Generate placeholder WAV files or short silence files for tests.

**Step 3: Add backend registry**

Define backend metadata for supported languages, voices, and license notes.

**Step 4: Run tests**

Run: `pytest workers/python/tests/test_tts.py -v`

Expected: TTS interface tests pass.

**Step 5: Commit**

```bash
git add workers/python
git commit -m "feat: add pluggable TTS backend interface"
```

### Task 12: Assemble Chapter Audio

**Files:**
- Create: `workers/python/audiobook_worker/audio.py`
- Create: `workers/python/tests/test_audio.py`

**Step 1: Write tests**

Use generated short segment files and assert that chapter assembly creates one output file.

**Step 2: Implement assembly**

Use ffmpeg or pydub to concatenate segment audio and add configurable silence gaps.

**Step 3: Run tests**

Run: `pytest workers/python/tests/test_audio.py -v`

Expected: chapter assembly tests pass.

**Step 4: Commit**

```bash
git add workers/python
git commit -m "feat: assemble chapter audio from segments"
```

## Phase 5: Desktop App

### Task 13: Scaffold Desktop App

**Files:**
- Create: `apps/desktop/`

**Step 1: Scaffold Tauri app**

Run the chosen Tauri TypeScript template.

**Step 2: Add basic screens**

Create screens for:

- book import
- job progress
- character summary
- chapter list
- export

**Step 3: Run app**

Run: `npm run dev --workspace apps/desktop`

Expected: desktop app opens.

**Step 4: Commit**

```bash
git add apps/desktop
git commit -m "feat: scaffold desktop application"
```

### Task 14: Add Local SQLite State

**Files:**
- Modify: `apps/desktop/`
- Create: `apps/desktop/src/state/`

**Step 1: Write state tests**

Test creating a book, chapter, job, character, voice, and artifact record.

**Step 2: Implement SQLite layer**

Add migrations and typed data access functions.

**Step 3: Run tests**

Run: `npm test --workspace apps/desktop`

Expected: state tests pass.

**Step 4: Commit**

```bash
git add apps/desktop
git commit -m "feat: add local audiobook state store"
```

### Task 15: Wire Worker Invocation

**Files:**
- Modify: `apps/desktop/src/`

**Step 1: Write worker invocation tests**

Mock the Python worker process and assert that JSON requests and responses are handled correctly.

**Step 2: Implement worker runner**

Add process invocation, timeout handling, log capture, and structured error handling.

**Step 3: Run tests**

Run: `npm test --workspace apps/desktop`

Expected: worker invocation tests pass.

**Step 4: Commit**

```bash
git add apps/desktop
git commit -m "feat: invoke python workers from desktop app"
```

### Task 16: Implement End-to-End MVP Flow

**Files:**
- Modify: `apps/desktop/src/`
- Modify: `workers/python/audiobook_worker/`

**Step 1: Add integration fixture**

Use a tiny public-domain or synthetic dialogue sample.

**Step 2: Run import through script generation**

Verify that the app can import a file, extract text, detect chapters, build a script, assign voices, and generate placeholder audio.

**Step 3: Replace placeholder TTS with first real backend**

Choose one local TTS backend and implement the minimum viable integration.

**Step 4: Generate one chapter**

Run a real local generation for one short chapter.

Expected: chapter audio is generated and playable.

**Step 5: Commit**

```bash
git add apps/desktop workers/python fixtures
git commit -m "feat: generate first local audiobook chapter"
```

## Phase 6: Review, Rights, and Polish

### Task 17: Add License and Rights Gate

**Files:**
- Create: `workers/python/audiobook_worker/rights.py`
- Create: `workers/python/tests/test_rights.py`
- Modify: `apps/desktop/src/`

**Step 1: Write rights classification tests**

Cover allowed, unknown, restricted, and blocked classifications.

**Step 2: Implement metadata inspection**

Inspect EPUB/PDF metadata and visible license text where available.

**Step 3: Add attestation UI**

Show user attestation when license is unknown or restricted.

**Step 4: Run tests**

Run Python and desktop tests.

Expected: rights tests pass and UI blocks only required cases.

**Step 5: Commit**

```bash
git add workers/python apps/desktop
git commit -m "feat: add license and rights gate"
```

### Task 18: Add Confidence-Based Review UI

**Files:**
- Modify: `apps/desktop/src/`

**Step 1: Write UI tests**

Test that low-confidence characters and chapters are surfaced.

**Step 2: Implement review screen**

Allow global corrections for aliases, gender, and voice assignment.

**Step 3: Implement regeneration trigger**

Regenerate affected scripts or chapters after corrections.

**Step 4: Run tests**

Run: `npm test --workspace apps/desktop`

Expected: review UI tests pass.

**Step 5: Commit**

```bash
git add apps/desktop
git commit -m "feat: add confidence-based review workflow"
```

## Completion Criteria

The MVP is complete when:

- A user can import a clean EPUB.
- A user can import a selectable-text PDF.
- The app detects chapters.
- The app creates a script IR with narration and dialogue.
- The app assigns voices automatically.
- The app generates at least one chapter using local TTS.
- The app exports playable chapter audio.
- The app stores resumable job state.
- Unknown or restricted license status requires user attestation.
- Low-confidence speaker or voice issues appear in optional review.
