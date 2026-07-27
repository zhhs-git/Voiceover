# Remove Mocks — LLM, Rights, Export, OCR Design

## Status

Draft.

## Summary

Replace hardcoded mocks and placeholders with real implementations: wire the LLM analyzer to use the configured DeepSeek API, make the rights gate actually block generation, add a working export flow, and implement OCR for scanned PDFs via Tesseract.

## 1. LLM — Wire Real Analysis

### Current state
`App.tsx:147` hardcodes `mockLlm: true`. The Python worker's `default_analyzer()` already reads from `~/.pi/agent/models.json` and picks up the DeepSeek provider with a valid API key.

### Changes
- Remove `mockLlm: true` from `App.tsx` so the worker uses `default_analyzer()`
- Add a checkbox toggle in the UI ("Use LLM analysis (slower, more accurate)") that maps to the `mockLlm` boolean — checked = `mockLlm: false` (use real LLM), unchecked = `mockLlm: true` (use heuristics)
- The `apply_corrections` command should also respect this toggle by passing through an `analyzer` param

### Data flow
User toggles LLM → `useLlm` boolean in state → passed as `mockLlm: !useLlm` to `analyze_chapter` and `apply_corrections` worker calls → `_analyze_chapter` in cli.py chooses `MockLLMAnalyzer` or `default_analyzer()`

## 2. Rights Gate

### Current state
`rights.py` has `classify_rights()` that inspects metadata/text for DRM, public domain, CC, and all-rights-reserved indicators. Not wired to CLI or UI. The rights panel in App.tsx just shows a checkbox that does nothing.

### Changes
- Add `check_rights` CLI command wrapping `classify_rights()`
- Call it during `handleImportBook` in App.tsx after extraction succeeds
- Store rights result in state: `{ classification, reason, requiresAttestation }`
- Block "Analyze" button when `classification === "blocked"` with a clear message
- Require attestation checkbox for `unknown`/`restricted` before enabling "Analyze"
- Show classification + reason in the Rights panel (replacing the static text)
- Store attestation timestamp in state

### CLI command
Input: `{ bookPath, metadata }` → Output: `{ status, classification, reason, requiresAttestation, evidence }`

## 3. Export

### Current state
The Export panel shows a code block with the file path. No save/export action.

### Changes
- Add a "Save Audio" button alongside the path display
- Use `@tauri-apps/plugin-dialog` `save()` to pick destination
- Copy the WAV file to the chosen location using Tauri's filesystem APIs
- Show success/failure message
- If multiple chapters were generated, allow saving individual chapters or all

## 4. OCR

### Current state
`ocr.py:33-36` raises `OCRBackendNotConfigured`. `extract.py:79-81` detects scanned/mixed PDFs but never runs OCR.

### Changes
- Install: `brew install tesseract` + `pip install pytesseract`
- Implement `run_ocr()` using pytesseract: iterate PDF pages, render each as image, run OCR, concatenate results
- Update `_extract_pdf()`: when `text_layer` is "scanned" or "mixed", call `run_ocr()` instead of `page.get_text("text")`
- For "mixed" PDFs, combine selectable text pages with OCR'd pages
- Track OCR confidence per page, surface as warnings
- Handle missing tesseract gracefully: if not installed, return `OCRBackendNotConfigured` as before

### Candidate libraries
- `pytesseract` for OCR
- `fitz` (PyMuPDF) for page-to-image rendering
- `PIL`/`Pillow` for image processing between PyMuPDF and pytesseract
