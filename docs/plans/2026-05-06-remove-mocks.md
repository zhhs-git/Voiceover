# Remove Mocks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded mocks and placeholders with real implementations — wire LLM analysis to DeepSeek API, make the rights gate functional, add audio export, and implement OCR for scanned PDFs.

**Architecture:** Python worker gets `check_rights` CLI command and real `run_ocr` with Tesseract. The desktop app adds an LLM toggle, calls `check_rights` during import, blocks the pipeline based on rights, and adds a file save dialog for export. All changes follow TDD with exact file paths.

**Tech Stack:** TypeScript (React/Tauri), Python (pytesseract/fitz), Tauri dialog plugin.

**Working directory:** `.worktrees/remove-mocks`

---

## Phase 1: Rights Gate

### Task 1: Add check_rights CLI command

**Files:**
- Modify: `workers/python/audiobook_worker/cli.py:80-97`
- Modify: `workers/python/tests/test_cli.py`

**Step 1: Write failing test**

In `workers/python/tests/test_cli.py`, add:

```python
def test_check_rights_classifies_allowed_public_domain(tmp_path: Path):
    from audiobook_worker.cli import main

    book_path = tmp_path / "test.txt"
    book_path.write_text("Project Gutenberg public domain work", encoding="utf-8")

    request = {
        "bookPath": str(book_path),
        "metadata": {"title": "Test Book"},
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["check_rights", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    assert result["classification"] == "allowed"
    assert result["reason"] == "public_domain_notice"
    assert result["requiresAttestation"] == False


def test_check_rights_classifies_blocked_drm(tmp_path: Path):
    from audiobook_worker.cli import main

    request = {
        "bookPath": str(tmp_path / "nonexistent.txt"),
        "metadata": {"drm": True},
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["check_rights", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["classification"] == "blocked"
    assert result["reason"] == "drm_detected"
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pytest tests/test_cli.py::test_check_rights_classifies_allowed_public_domain tests/test_cli.py::test_check_rights_classifies_blocked_drm -v
```

Expected: FAIL — unknown_command

**Step 3: Add dispatch and handler in cli.py**

Add to `_dispatch` (after the last `if command ==` block):
```python
    if command == "check_rights":
        return _check_rights(request)
```

Add handler:
```python
def _check_rights(request: dict[str, Any]) -> dict[str, Any]:
    from audiobook_worker.rights import classify_rights

    result = classify_rights(
        input_path=Path(request["bookPath"]),
        metadata=request.get("metadata", {}),
    )
    payload = _response("succeeded")
    payload["classification"] = result.classification
    payload["reason"] = result.reason
    payload["requiresAttestation"] = result.requires_attestation
    payload["evidence"] = result.evidence
    return payload
```

Add `from pathlib import Path` at top of cli.py if not already imported (it already is at line 5).

**Step 4: Run tests to verify they pass**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pytest tests/test_cli.py::test_check_rights_classifies_allowed_public_domain tests/test_cli.py::test_check_rights_classifies_blocked_drm -v
```

Expected: 2 PASS

**Step 5: Run full test suite**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pytest -v
```

Expected: all 45 tests pass (43 existing + 2 new).

**Step 6: Commit**

```bash
git add workers/python/audiobook_worker/cli.py workers/python/tests/test_cli.py
git commit -m "feat: add check_rights CLI command"
```

---

### Task 2: Wire rights into App.tsx

**Files:**
- Modify: `apps/desktop/src/App.tsx:1-449`

**Step 1: Update App.test.tsx to test rights state**

Replace `apps/desktop/src/App.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { App } from "./App";

describe("App", () => {
  test("shows the core MVP workflow screens", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Audiobook Generator" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Import Book" })).toBeInTheDocument();
    expect(screen.getByText("Job Progress")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Characters" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Chapters" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Rights" })).toBeInTheDocument();
    expect(screen.getByLabelText("I have the right to convert this book")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Review" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Export" })).toBeInTheDocument();
  });

  test("shows blocked message when rights classification is blocked", () => {
    render(<App />);

    // No book loaded → shows placeholder text
    expect(screen.getByText("Unknown or restricted license status will require confirmation before generation.")).toBeInTheDocument();
  });

  test("review panel shows placeholder text when no analysis loaded", () => {
    render(<App />);
    expect(screen.getByText("Run analysis first to see the character table and make corrections.")).toBeInTheDocument();
  });
});
```

**Step 2: Run tests to verify they fail**

```bash
cd .worktrees/remove-mocks/apps/desktop
npm test -- --run
```

Expected: FAIL — rights panel shows old text

**Step 3: Modify App.tsx — add rights state and LLM toggle**

Changes to `App.tsx`:

1. Add interfaces:
```typescript
interface RightsResult {
  classification: string;
  reason: string;
  requiresAttestation: boolean;
  evidence: string[];
}
```

2. Add state variables:
```typescript
const [rights, setRights] = useState<RightsResult | null>(null);
const [rightsAttested, setRightsAttested] = useState(false);
const [useLlm, setUseLlm] = useState(false);
```

3. In `handleImportBook`, after extraction succeeds, add rights check:
```typescript
      // Check rights
      try {
        const rightsResult = await workerCall("check_rights", {
          bookPath: path,
          metadata: {},
        });
        if (rightsResult.status === "succeeded") {
          setRights({
            classification: rightsResult.classification as string,
            reason: rightsResult.reason as string,
            requiresAttestation: rightsResult.requiresAttestation as boolean,
            evidence: rightsResult.evidence as string[],
          });
        }
      } catch {
        setRights({ classification: "unknown", reason: "check_failed", requiresAttestation: true, evidence: [] });
      }
```

4. Reset rights on new import:
```typescript
    setRights(null);
    setRightsAttested(false);
```

5. Change `mockLlm: true` → `mockLlm: !useLlm` in `analyze_chapter` call.

6. Add rights-based blocking logic:
- "Analyze Book" button only shows when `!book || rights.classification !== "blocked"`
- If `blocked`, show "Cannot analyze: DRM detected" in review panel
- If `requiresAttestation && !rightsAttested`, "Analyze Book" is disabled

7. Update Rights panel to show real data:
```tsx
          <article>
            <h3>Rights</h3>
            {rights ? (
              <>
                <p className={`rights-badge rights-${rights.classification}`}>
                  {rights.classification.toUpperCase()}
                  {rights.classification === "blocked" && " — Cannot proceed"}
                </p>
                <p className="rights-reason">{rights.reason.replace(/_/g, " ")}</p>
                {rights.requiresAttestation && (
                  <label className="attestation">
                    <input
                      type="checkbox"
                      checked={rightsAttested}
                      onChange={(e) => setRightsAttested(e.target.checked)}
                    />
                    <span>I have the right to convert this book</span>
                  </label>
                )}
              </>
            ) : (
              <>
                <p>Unknown or restricted license status will require confirmation before generation.</p>
                <label className="attestation">
                  <input type="checkbox" disabled />
                  <span>I have the right to convert this book</span>
                </label>
              </>
            )}
          </article>
```

8. Add LLM toggle in sidebar (after import button):
```tsx
        {book && (
          <label className="llm-toggle" style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, fontSize: "0.875rem" }}>
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            <span>Use LLM analysis (slower, more accurate)</span>
          </label>
        )}
```

9. Update the sidebar "Analyze Book" button condition:
```tsx
        {book && !analysis && rights?.classification !== "blocked" && (
          <button
            className="primary-action"
            type="button"
            onClick={handleAnalyze}
            disabled={stage === "analyzing" || (rights?.requiresAttestation && !rightsAttested)}
            style={{ marginTop: 8 }}
          >
            {stage === "analyzing" ? "Analyzing..." : rights?.requiresAttestation && !rightsAttested ? "Attest rights first" : "Analyze Book"}
          </button>
        )}
```

10. In `handleSaveCorrections`, pass `mockLlm: !useLlm`:
```typescript
      const result = await workerCall("apply_corrections", {
        bookId: book.bookId,
        chapters: chaptersInput,
        corrections: {
          aliasMerges: correctionState.aliasMerges,
          genderOverrides: correctionState.genderOverrides,
          voiceOverrides: correctionState.voiceOverrides,
        },
        outputDirectory: `${book.workDir}/scripts`,
        language: "en",
        mockLlm: !useLlm,
      });
```

11. Also update `apply_corrections` CLI handler to respect `mockLlm` flag. In `cli.py`, change `_apply_corrections` to pass analyzer:
```python
def _apply_corrections(request: dict[str, Any]) -> dict[str, Any]:
    output_directory = Path(request["outputDirectory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    corrections = request.get("corrections", {})
    analyzer = MockLLMAnalyzer() if request.get("mockLlm") else default_analyzer()

    artifacts = []
    for chapter in request["chapters"]:
        chapter_text = Path(chapter["textPath"]).read_text(encoding="utf-8")
        script = build_chapter_script_with_corrections(
            book_id=request["bookId"],
            chapter_id=chapter["chapterId"],
            title=chapter.get("title", chapter["chapterId"]),
            text=chapter_text,
            language=request.get("language", "en"),
            corrections=corrections,
            analyzer=analyzer,
        )
        # ... rest unchanged
```

**Step 4: Run full test suite**

```bash
cd .worktrees/remove-mocks/apps/desktop && npm test -- --run
cd .worktrees/remove-mocks/workers/python && .venv/bin/pytest -v
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add apps/desktop/src/App.tsx apps/desktop/src/App.test.tsx workers/python/audiobook_worker/cli.py
git commit -m "feat: wire rights gate and LLM toggle into app"
```

---

## Phase 2: Export

### Task 3: Add export functionality

**Files:**
- Modify: `apps/desktop/src/App.tsx` (the Export article section)
- Modify: `apps/desktop/src/styles.css` (export styles)

**Step 1: Update Export panel in App.tsx**

Replace the Export article (around line 440-448) with:

```tsx
          <article>
            <h3>Export</h3>
            {audioPath ? (
              <>
                <p>Chapter audio ready:</p>
                <code className="export-path">{audioPath}</code>
                <button
                  className="primary-action"
                  type="button"
                  onClick={async () => {
                    try {
                      const savePath = await open({
                        multiple: false,
                        defaultPath: "chapter.wav",
                        filters: [{ name: "Audio", extensions: ["wav"] }],
                      });
                      if (!savePath) return;
                      await invoke("copy_file", { from: audioPath, to: savePath as string });
                      setSavedMessage(`Saved to ${savePath}`);
                    } catch (err) {
                      setError(String(err));
                    }
                  }}
                  style={{ marginTop: 12 }}
                >
                  Save Audio File
                </button>
              </>
            ) : (
              <p>Completed chapter audio and metadata exports will be available after generation.</p>
            )}
          </article>
```

Note: We need a `copy_file` Tauri command. Create or update `apps/desktop/src-tauri/src/lib.rs`:

Add:
```rust
#[tauri::command]
fn copy_file(from: String, to: String) -> Result<String, String> {
    std::fs::copy(&from, &to).map_err(|e| e.to_string())?;
    Ok(to)
}
```

And register it:
```rust
        .invoke_handler(tauri::generate_handler![run_worker, copy_file])
```

**Step 2: Add CSS for export path**

In `apps/desktop/src/styles.css`, add:
```css
.export-path {
  display: block;
  word-break: break-all;
  padding: 8px 10px;
  margin: 8px 0;
  background: #f1f5f9;
  border: 1px solid #dfe5ec;
  border-radius: 6px;
  font-size: 0.75rem;
  color: #475569;
}

.rights-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 6px;
}

.rights-allowed { background: rgba(34, 197, 94, 0.1); color: #166534; }
.rights-restricted { background: rgba(234, 179, 8, 0.1); color: #854d0e; }
.rights-unknown { background: rgba(148, 163, 184, 0.1); color: #475569; }
.rights-blocked { background: rgba(239, 68, 68, 0.1); color: #991b1b; }

.rights-reason {
  font-size: 0.8125rem;
  color: #66717f;
  text-transform: capitalize;
}
```

**Step 3: Run tests**

```bash
cd .worktrees/remove-mocks/apps/desktop && npm test -- --run
cd .worktrees/remove-mocks/workers/python && .venv/bin/pytest -v
```

**Step 4: Commit**

```bash
git add apps/desktop/src/App.tsx apps/desktop/src/styles.css apps/desktop/src-tauri/src/lib.rs
git commit -m "feat: add export save dialog and copy_file Tauri command"
```

---

## Phase 3: OCR

### Task 4: Install OCR dependencies

**Step 1: Install system-level tesseract**

```bash
brew install tesseract
```

**Step 2: Install Python OCR package**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pip install pytesseract Pillow
```

**Step 3: Verify installation**

```bash
.venv/bin/python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Expected: prints tesseract version (e.g. `5.x.x`)

**Step 4: Commit**

```bash
git add workers/python/pyproject.toml
git commit -m "chore: add pytesseract and Pillow OCR dependencies"
```

---

### Task 5: Implement real OCR

**Files:**
- Modify: `workers/python/audiobook_worker/ocr.py:1-37`
- Modify: `workers/python/audiobook_worker/extract.py:60-81`
- Modify: `workers/python/tests/test_ocr_detection.py`

**Step 1: Write failing test for OCR output**

Replace `test_placeholder_ocr_backend_returns_clear_error` in `test_ocr_detection.py`:

```python
def test_run_ocr_returns_text_from_image(tmp_path: Path):
    """OCR extracts text from a rendered PDF page."""
    from audiobook_worker.ocr import run_ocr
    import fitz

    # Create a tiny PDF with selectable text, then OCR it anyway
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text(fitz.Point(20, 50), "Hello World", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    text = run_ocr(str(pdf_path))
    assert "Hello" in text or "World" in text
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pytest tests/test_ocr_detection.py::test_run_ocr_returns_text_from_image -v
```

Expected: FAIL — `OCRBackendNotConfigured`

**Step 3: Implement real OCR in ocr.py**

Replace `run_ocr()`:

```python
def run_ocr(input_path: Path | str) -> str:
    path = Path(input_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"OCR only supports PDF files, got: {path.suffix}")

    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise OCRBackendNotConfigured(
            "pytesseract is not installed. Run: pip install pytesseract Pillow"
        )

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise OCRBackendNotConfigured(
            "Tesseract is not installed. Run: brew install tesseract"
        )

    document = fitz.open(path)
    page_texts: list[str] = []

    try:
        for page in document:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            normalized = " ".join(text.split())
            if normalized:
                page_texts.append(normalized)
    finally:
        document.close()

    if not page_texts:
        raise OCRBackendNotConfigured(
            f"No text could be extracted from {path.name}"
        )

    return "\n\n".join(page_texts)
```

**Step 4: Update _extract_pdf to use OCR for scanned PDFs**

In `extract.py`, replace `_extract_pdf`:

```python
def _extract_pdf(path: Path) -> ExtractedBookText:
    from audiobook_worker.ocr import run_ocr

    text_layer = classify_pdf_text_layer(path)
    document = fitz.open(path)
    page_texts: list[str] = []
    ocr_warnings: list[str] = []
    try:
        if text_layer == "scanned":
            try:
                text = run_ocr(path)
                page_texts.append(text)
            except Exception as e:
                return ExtractedBookText(
                    kind="pdf",
                    text="",
                    metadata={
                        "page_count": document.page_count,
                        "text_layer": text_layer,
                        **{key: value for key, value in document.metadata.items() if value},
                    },
                    requires_ocr=True,
                    warnings=[f"OCR failed: {e}"],
                )
        elif text_layer == "mixed":
            for page in document:
                page_text = page.get_text("text").strip()
                if page_text:
                    page_texts.append(_normalize_text(page_text))
                else:
                    ocr_warnings.append("mixed_page_required_ocr")
            if ocr_warnings:
                ocr_warnings.append("Consider re-running with full OCR for best results")
        else:
            for page in document:
                page_texts.append(page.get_text("text"))

        metadata = {
            "page_count": document.page_count,
            "text_layer": text_layer,
            **{key: value for key, value in document.metadata.items() if value},
        }
    finally:
        document.close()

    text = _normalize_text("\n\n".join(page_texts))
    return ExtractedBookText(
        kind="pdf",
        text=text,
        metadata=metadata,
        requires_ocr=text_layer in {"scanned", "mixed"},
        warnings=ocr_warnings if ocr_warnings else (
            ["requires_ocr"] if text_layer in {"scanned", "mixed"} else []
        ),
    )
```

**Step 5: Add missing import in extract.py**

At the top of `extract.py`, ensure `_normalize_text` is already defined (it is at line 84).

**Step 6: Run tests**

```bash
cd .worktrees/remove-mocks/workers/python
.venv/bin/pytest tests/test_ocr_detection.py -v
.venv/bin/pytest -v
```

Expected: all OCR tests + all other tests pass.

**Step 7: Commit**

```bash
git add workers/python/audiobook_worker/ocr.py workers/python/audiobook_worker/extract.py workers/python/tests/test_ocr_detection.py
git commit -m "feat: implement real OCR with pytesseract for scanned PDFs"
```

---

### Task 6: Update pyproject.toml with new deps

**Files:**
- Modify: `workers/python/pyproject.toml`

**Step 1: Add pytesseract and Pillow to dependencies**

Add to `[project]` `dependencies` list:
```toml
  "pytesseract>=0.3.10",
  "Pillow>=10.0.0",
```

**Step 2: Commit**

```bash
git add workers/python/pyproject.toml
git commit -m "chore: declare pytesseract and Pillow in pyproject.toml"
```

---

## Completion Criteria

- [ ] `check_rights` CLI command works and returns classification
- [ ] Rights panel shows real classification, blocks pipeline on blocked/DRM
- [ ] Attestation checkbox gates the Analyze button for unknown/restricted
- [ ] LLM toggle exists and switches between mock (heuristics) and real (DeepSeek)
- [ ] Export button opens native save dialog and copies audio file
- [ ] `run_ocr` uses pytesseract to extract text from scanned PDF page images
- [ ] `_extract_pdf` calls OCR for scanned PDFs, falls back to selectable text otherwise
- [ ] All existing tests pass (Python + TypeScript)
- [ ] All new tests pass
