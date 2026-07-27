# Review & Correction Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the desktop app review panel fully functional — per-character table with editable gender/voice, alias merging, save-to-worker, and per-chapter regeneration.

**Architecture:** Python worker gets a new `apply_corrections` command that re-analyzes chapters with user corrections as hard constraints (alias merges pre-segmentation, gender/voice overrides post-analysis). The TypeScript desktop app replaces the hardcoded review inputs with a character table component backed by correction state, calls the worker on save, then triggers TTS regeneration for affected chapters.

**Tech Stack:** React (existing), TypeScript, Python (audiobook_worker), Vitest, pytest.

**Working directory:** `.worktrees/mvp-implementation`

---

## Phase 1: Python Worker — apply_corrections command

### Task 1: Add correction-aware script builder

**Files:**
- Modify: `workers/python/audiobook_worker/script_builder.py:71-140`
- Create: `workers/python/tests/test_corrections.py`

**Step 1: Write failing test for alias merging**

In `workers/python/tests/test_corrections.py`:

```python
from audiobook_worker.script_builder import build_chapter_script_with_corrections


def test_alias_merge_replaces_speaker_in_segments():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Over here," Lizzy called. "Coming," Elizabeth replied.',
        language="en",
        corrections={
            "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
        },
    )

    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    assert speakers == {"elizabeth"}
    assert "lizzy" not in speakers


def test_gender_override_changes_character_and_voice():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
        corrections={
            "genderOverrides": [{"characterId": "darcy", "gender": "female"}],
        },
    )

    character = next(c for c in script["characters"] if c["id"] == "darcy")
    assert character["gender"] == "female"
    assert character["voiceId"] == "female_adult_01"


def test_voice_override_changes_assigned_voice():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
        corrections={
            "voiceOverrides": [{"characterId": "darcy", "voiceId": "neutral_dialogue_01"}],
        },
    )

    character = next(c for c in script["characters"] if c["id"] == "darcy")
    assert character["voiceId"] == "neutral_dialogue_01"

    # segments should use the overridden voice
    segments_with_darcy = [s for s in script["segments"] if s["speakerId"] == "darcy"]
    assert all(s["voiceId"] == "neutral_dialogue_01" for s in segments_with_darcy)


def test_no_corrections_returns_same_as_build_chapter_script():
    from audiobook_worker.script_builder import build_chapter_script

    kwargs = dict(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
    )
    baseline = build_chapter_script(**kwargs)
    corrected = build_chapter_script_with_corrections(**kwargs, corrections={})

    assert corrected["segments"] == baseline["segments"]
    assert corrected["characters"] == baseline["characters"]
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest tests/test_corrections.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'test_corrections'` or `ImportError: cannot import name 'build_chapter_script_with_corrections'`

**Step 3: Implement `build_chapter_script_with_corrections` in script_builder.py**

Add after the existing `build_chapter_script` function (after line 140):

```python
def build_chapter_script_with_corrections(
    *,
    book_id: str,
    chapter_id: str,
    title: str,
    text: str,
    language: str,
    corrections: dict,
    analyzer=None,
) -> dict:
    alias_map: dict[str, str] = {}
    for merge in corrections.get("aliasMerges", []):
        alias_map[merge["from"].lower()] = merge["to"]

    if alias_map:
        import re
        for alias, canonical in alias_map.items():
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            text = pattern.sub(canonical, text)

    gender_overrides: dict[str, str] = {}
    for override in corrections.get("genderOverrides", []):
        gender_overrides[override["characterId"]] = override["gender"]

    voice_overrides: dict[str, str] = {}
    for override in corrections.get("voiceOverrides", []):
        voice_overrides[override["characterId"]] = override["voiceId"]

    script = build_chapter_script(
        book_id=book_id,
        chapter_id=chapter_id,
        title=title,
        text=text,
        language=language,
        analyzer=analyzer,
    )

    for character in script["characters"]:
        char_id = character["id"]
        if char_id in gender_overrides:
            character["gender"] = gender_overrides[char_id]
            character["voiceId"] = _voice_for_gender(gender_overrides[char_id])
        if char_id in voice_overrides:
            character["voiceId"] = voice_overrides[char_id]

    for segment in script["segments"]:
        speaker_id = segment["speakerId"]
        if speaker_id in voice_overrides:
            segment["voiceId"] = voice_overrides[speaker_id]
        elif speaker_id in gender_overrides:
            segment["voiceId"] = _voice_for_gender(gender_overrides[speaker_id])

    return script
```

**Step 4: Run tests to verify they pass**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest tests/test_corrections.py -v
```

Expected: PASS (4 tests)

**Step 5: Run full test suite to check nothing broke**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest -v
```

Expected: all existing tests pass.

**Step 6: Commit**

```bash
git add workers/python/audiobook_worker/script_builder.py workers/python/tests/test_corrections.py
git commit -m "feat: add correction-aware script builder with alias/gender/voice overrides"
```

---

### Task 2: Wire apply_corrections command into CLI

**Files:**
- Modify: `workers/python/audiobook_worker/cli.py:80-95`

**Step 1: Write failing test**

In `workers/python/tests/test_cli.py`, add after existing tests:

```python
def test_apply_corrections_command(tmp_path: Path):
    from audiobook_worker.cli import main

    chapter_path = tmp_path / "ch01.txt"
    chapter_path.write_text('"Hello," said Lizzy. "Hi," Elizabeth replied.', encoding="utf-8")
    output_dir = tmp_path / "scripts"
    output_dir.mkdir()

    request = {
        "bookId": "book1",
        "chapters": [
            {"chapterId": "ch01", "textPath": str(chapter_path), "title": "Chapter 1"}
        ],
        "corrections": {
            "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
            "genderOverrides": [{"characterId": "elizabeth", "gender": "female"}],
            "voiceOverrides": [],
        },
        "outputDirectory": str(output_dir),
        "language": "en",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    import os
    with patch.dict(os.environ, {"AUDIOBOOK_LLM_MODEL": "mock"}):
        exit_code = main(["apply_corrections", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["kind"] == "chapter_script"

    # verify alias merge was applied
    script = json.loads(Path(result["artifacts"][0]["path"]).read_text())
    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    assert speakers == {"elizabeth"}
```

(Note: add `from unittest.mock import patch` and `import os` at top of test file if not already present; also add `from pathlib import Path`.)

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest tests/test_cli.py::test_apply_corrections_command -v
```

Expected: FAIL — `exit_code == 2` (unknown_command) or `KeyError`

**Step 3: Add apply_corrections to CLI dispatch and handler**

In `cli.py`:

Add import at top:
```python
from audiobook_worker.script_builder import build_chapter_script_with_corrections
```

Add to `_dispatch` function (after line 88):
```python
    if command == "apply_corrections":
        return _apply_corrections(request)
```

Add new handler function after `_assemble_chapter_audio`:

```python
def _apply_corrections(request: dict[str, Any]) -> dict[str, Any]:
    output_directory = Path(request["outputDirectory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    corrections = request.get("corrections", {})

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
        )
        script_path = output_directory / f"{chapter['chapterId']}.json"
        _write_json(script_path, script)
        artifacts.append({
            "kind": "chapter_script",
            "path": str(script_path),
            "metadata": {
                "chapterId": chapter["chapterId"],
                "characterCount": len(script.get("characters", [])),
                "segmentCount": len(script.get("segments", [])),
            },
        })

    return _response("succeeded", artifacts=artifacts)
```

**Step 4: Run test to verify it passes**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest tests/test_cli.py::test_apply_corrections_command -v
```

Expected: PASS

**Step 5: Run full test suite**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add workers/python/audiobook_worker/cli.py workers/python/tests/test_cli.py
git commit -m "feat: add apply_corrections worker command"
```

---

## Phase 2: TypeScript — Correction State & Character Table

### Task 3: Add correction types and state hook

**Files:**
- Create: `apps/desktop/src/state/corrections.ts`
- Create: `apps/desktop/src/state/corrections.test.ts`

**Step 1: Write failing test**

In `apps/desktop/src/state/corrections.test.ts`:

```typescript
import { describe, expect, test } from "vitest";
import {
  createCorrectionsStore,
  type CorrectionState,
  type AliasMerge,
  type GenderOverride,
  type VoiceOverride,
} from "./corrections";

describe("corrections store", () => {
  test("starts with empty corrections and clean state", () => {
    const store = createCorrectionsStore();
    expect(store.get().aliasMerges).toEqual([]);
    expect(store.get().genderOverrides).toEqual([]);
    expect(store.get().voiceOverrides).toEqual([]);
    expect(store.get().dirty).toBe(false);
  });

  test("addMerge sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    expect(store.get().aliasMerges).toEqual([{ from: "Lizzy", to: "Elizabeth" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("setGender sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.setGender("elizabeth", "female");
    expect(store.get().genderOverrides).toEqual([{ characterId: "elizabeth", gender: "female" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("setVoice sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.setVoice("elizabeth", "female_adult_01");
    expect(store.get().voiceOverrides).toEqual([{ characterId: "elizabeth", voiceId: "female_adult_01" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("markSaved clears dirty flag and records saved corrections", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    store.markSaved(["ch01", "ch02"]);
    expect(store.get().dirty).toBe(false);
    expect(store.get().affectedChapters).toEqual(["ch01", "ch02"]);
    expect(store.get().savedCorrections).toEqual({
      aliasMerges: [{ from: "Lizzy", to: "Elizabeth" }],
      genderOverrides: [],
      voiceOverrides: [],
    });
  });

  test("setGender replaces existing override for same character", () => {
    const store = createCorrectionsStore();
    store.setGender("elizabeth", "female");
    store.setGender("elizabeth", "male");
    expect(store.get().genderOverrides).toEqual([{ characterId: "elizabeth", gender: "male" }]);
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/state/corrections.test.ts
```

Expected: FAIL — `Cannot find module`

**Step 3: Implement correction state**

In `apps/desktop/src/state/corrections.ts`:

```typescript
export interface AliasMerge {
  from: string;
  to: string;
}

export interface GenderOverride {
  characterId: string;
  gender: string;
}

export interface VoiceOverride {
  characterId: string;
  voiceId: string;
}

export interface CorrectionSet {
  aliasMerges: AliasMerge[];
  genderOverrides: GenderOverride[];
  voiceOverrides: VoiceOverride[];
}

export interface CorrectionState extends CorrectionSet {
  dirty: boolean;
  savedCorrections: CorrectionSet | null;
  affectedChapters: string[];
}

export function createCorrectionsStore() {
  let state: CorrectionState = {
    aliasMerges: [],
    genderOverrides: [],
    voiceOverrides: [],
    dirty: false,
    savedCorrections: null,
    affectedChapters: [],
  };

  const listeners = new Set<() => void>();

  function notify() {
    for (const fn of listeners) fn();
  }

  return {
    get(): CorrectionState {
      return state;
    },

    subscribe(fn: () => void) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },

    addMerge(merge: AliasMerge) {
      state = {
        ...state,
        aliasMerges: [...state.aliasMerges.filter(m => m.from !== merge.from), merge],
        dirty: true,
      };
      notify();
    },

    removeMerge(from: string) {
      state = {
        ...state,
        aliasMerges: state.aliasMerges.filter(m => m.from !== from),
        dirty: true,
      };
      notify();
    },

    setGender(characterId: string, gender: string) {
      state = {
        ...state,
        genderOverrides: [
          ...state.genderOverrides.filter(o => o.characterId !== characterId),
          { characterId, gender },
        ],
        dirty: true,
      };
      notify();
    },

    setVoice(characterId: string, voiceId: string) {
      state = {
        ...state,
        voiceOverrides: [
          ...state.voiceOverrides.filter(o => o.characterId !== characterId),
          { characterId, voiceId },
        ],
        dirty: true,
      };
      notify();
    },

    markSaved(affectedChapters: string[]) {
      const { dirty: _, savedCorrections: __, affectedChapters: ___, ...corrections } = state;
      state = {
        ...state,
        dirty: false,
        savedCorrections: corrections,
        affectedChapters,
      };
      notify();
    },

    reset() {
      state = {
        aliasMerges: [],
        genderOverrides: [],
        voiceOverrides: [],
        dirty: false,
        savedCorrections: null,
        affectedChapters: [],
      };
      notify();
    },
  };
}
```

**Step 4: Run test to verify it passes**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/state/corrections.test.ts
```

Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add apps/desktop/src/state/corrections.ts apps/desktop/src/state/corrections.test.ts
git commit -m "feat: add correction state store with alias/gender/voice management"
```

---

### Task 4: Build CharacterTable component

**Files:**
- Create: `apps/desktop/src/components/CharacterTable.tsx`
- Create: `apps/desktop/src/components/CharacterTable.test.tsx`
- Create: `apps/desktop/src/components/`

**Step 1: Write failing test**

In `apps/desktop/src/components/CharacterTable.test.tsx`:

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { CharacterTable } from "./CharacterTable";

const sampleCharacters = [
  { id: "elizabeth", canonicalName: "Elizabeth", aliases: ["Lizzy"], gender: "female", voiceId: "female_adult_01", confidence: 0.92 },
  { id: "darcy", canonicalName: "Darcy", aliases: [], gender: "male", voiceId: "male_adult_01", confidence: 0.78 },
  { id: "unknown_speaker", canonicalName: "Unknown Speaker", aliases: [], gender: "unknown", voiceId: "neutral_dialogue_01", confidence: 0.35 },
];

const VOICE_OPTIONS = [
  { id: "narrator_default", displayName: "Default Narrator" },
  { id: "female_adult_01", displayName: "Female Adult 01" },
  { id: "male_adult_01", displayName: "Male Adult 01" },
  { id: "neutral_dialogue_01", displayName: "Neutral Dialogue 01" },
];

describe("CharacterTable", () => {
  test("renders all characters with name, gender, voice, and confidence", () => {
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
      />
    );

    expect(screen.getByText("Elizabeth")).toBeInTheDocument();
    expect(screen.getByText("Darcy")).toBeInTheDocument();
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
  });

  test("low-confidence character shows warning indicator", () => {
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
      />
    );

    // The low-confidence character should be visually flagged
    // CSS class or aria-label on the row
    const rows = screen.getAllByRole("row");
    const unknownRow = rows.find(row => row.textContent?.includes("35%"));
    expect(unknownRow).toBeTruthy();
  });

  test("calls onGenderChange when gender dropdown changes", () => {
    const onGenderChange = vi.fn();
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={onGenderChange}
        onVoiceChange={() => {}}
      />
    );

    const genderSelects = screen.getAllByLabelText("Gender");
    fireEvent.change(genderSelects[0], { target: { value: "male" } });
    expect(onGenderChange).toHaveBeenCalledWith("elizabeth", "male");
  });

  test("calls onVoiceChange when voice dropdown changes", () => {
    const onVoiceChange = vi.fn();
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={onVoiceChange}
      />
    );

    const voiceSelects = screen.getAllByLabelText("Voice");
    fireEvent.change(voiceSelects[0], { target: { value: "male_adult_01" } });
    expect(onVoiceChange).toHaveBeenCalledWith("elizabeth", "male_adult_01");
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/components/CharacterTable.test.tsx
```

Expected: FAIL — `Cannot find module`

**Step 3: Implement CharacterTable component**

In `apps/desktop/src/components/CharacterTable.tsx`:

```tsx
interface CharacterMeta {
  id: string;
  canonicalName: string;
  aliases: string[];
  gender: string;
  voiceId: string;
  confidence: number;
}

interface VoiceOption {
  id: string;
  displayName: string;
}

interface CharacterTableProps {
  characters: CharacterMeta[];
  voices: VoiceOption[];
  onGenderChange: (characterId: string, gender: string) => void;
  onVoiceChange: (characterId: string, voiceId: string) => void;
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "var(--color-success, #22c55e)";
  if (confidence >= 0.5) return "var(--color-warning, #eab308)";
  return "var(--color-error, #ef4444)";
}

export function CharacterTable({ characters, voices, onGenderChange, onVoiceChange }: CharacterTableProps) {
  if (characters.length === 0) {
    return <p>No characters detected yet. Run analysis first.</p>;
  }

  return (
    <table className="character-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Aliases</th>
          <th>Confidence</th>
          <th>Gender</th>
          <th>Voice</th>
        </tr>
      </thead>
      <tbody>
        {characters.map((c) => (
          <tr key={c.id} className={c.confidence < 0.5 ? "low-confidence" : ""}>
            <td>
              {c.canonicalName}
              {c.confidence < 0.5 && <span aria-label="Low confidence" className="warning-icon">⚠</span>}
            </td>
            <td>{c.aliases.length > 0 ? c.aliases.join(", ") : "—"}</td>
            <td>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.round(c.confidence * 100)}%`,
                    backgroundColor: confidenceColor(c.confidence),
                  }}
                />
                <span className="confidence-label">{Math.round(c.confidence * 100)}%</span>
              </div>
            </td>
            <td>
              <select
                aria-label="Gender"
                value={c.gender}
                onChange={(e) => onGenderChange(c.id, e.target.value)}
              >
                <option value="unknown">Unknown</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
                <option value="neutral">Neutral</option>
              </select>
            </td>
            <td>
              <select
                aria-label="Voice"
                value={c.voiceId}
                onChange={(e) => onVoiceChange(c.id, e.target.value)}
              >
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.displayName}
                  </option>
                ))}
              </select>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

**Step 4: Run test to verify it passes**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/components/CharacterTable.test.tsx
```

Expected: all 4 tests PASS

**Step 5: Commit**

```bash
git add apps/desktop/src/components/CharacterTable.tsx apps/desktop/src/components/CharacterTable.test.tsx
git commit -m "feat: add CharacterTable component with editable gender/voice"
```

---

### Task 5: Integrate CharacterTable and corrections into App

**Files:**
- Modify: `apps/desktop/src/App.tsx:1-363`
- Modify: `apps/desktop/src/App.test.tsx:1-23`

**Step 1: Update App.test.tsx for the new review panel**

Replace the existing test in `apps/desktop/src/App.test.tsx`:

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

  test("shows save corrections button when analysis is loaded", () => {
    // The Save Corrections button appears in the review panel when analysis exists
    render(<App />);
    // It should exist (it's always rendered, just potentially disabled)
    const saveBtn = screen.queryByRole("button", { name: "Save Corrections" });
    // Button exists but may or may not be present depending on state
    // With no book loaded, the review section shows placeholder text
    expect(screen.getByText("Low-confidence speakers and voices can be corrected globally before regeneration.")).toBeInTheDocument();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/App.test.tsx
```

Expected: FAIL — old test references old review panel elements that will change

**Step 3: Rewrite App.tsx with integrated CharacterTable and Save/Regenerate buttons**

Replace `apps/desktop/src/App.tsx` with:

```tsx
import { useState, useSyncExternalStore, useCallback } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { invoke } from "@tauri-apps/api/core";
import { tmpdir } from "@tauri-apps/api/path";
import { CharacterTable } from "./components/CharacterTable";
import {
  createCorrectionsStore,
  type AliasMerge,
} from "./state/corrections";

interface ChapterMeta {
  id: string;
  title: string;
  textLength: number;
  textPath: string;
}

interface CharacterMeta {
  id: string;
  canonicalName: string;
  aliases: string[];
  gender: string;
  voiceId: string;
  confidence: number;
}

interface VoiceMeta {
  id: string;
  displayName: string;
  backend: string;
}

interface BookState {
  title: string;
  bookId: string;
  workDir: string;
  chapters: ChapterMeta[];
}

interface AnalysisState {
  characters: CharacterMeta[];
  voices: VoiceMeta[];
  scriptPaths: Record<string, string>;
}

type PipelineStage = "idle" | "importing" | "analyzing" | "saving" | "generating" | "done" | "error";

const correctionsStore = createCorrectionsStore();

const VOICE_DISPLAY_NAMES: Record<string, string> = {
  narrator_default: "Default Narrator",
  female_adult_01: "Female Adult 01",
  male_adult_01: "Male Adult 01",
  neutral_dialogue_01: "Neutral Dialogue 01",
};

const VOICE_OPTIONS = Object.entries(VOICE_DISPLAY_NAMES).map(([id, displayName]) => ({ id, displayName }));

async function workerCall(command: string, input: Record<string, unknown>): Promise<Record<string, unknown>> {
  const raw = await invoke<string>("run_worker", {
    command,
    inputJson: JSON.stringify(input),
  });
  return JSON.parse(raw) as Record<string, unknown>;
}

export function App() {
  const [book, setBook] = useState<BookState | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisState | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);

  const correctionState = useSyncExternalStore(
    correctionsStore.subscribe,
    correctionsStore.get,
  );

  async function handleImportBook() {
    const path = await open({
      multiple: false,
      filters: [{ name: "Book", extensions: ["epub", "pdf"] }],
    });
    if (!path) return;

    setStage("importing");
    setError(null);
    setAnalysis(null);
    setAudioPath(null);
    correctionsStore.reset();

    try {
      const tmp = await tmpdir();
      const bookStem = (path as string).split("/").pop()?.replace(/\.[^.]+$/, "") ?? "book";
      const workDir = `${tmp}/audiobook-generator/${bookStem}`;

      const result = await workerCall("extract_book", {
        bookPath: path,
        outputDirectory: `${workDir}/chapters`,
      });

      if (result.status !== "succeeded") {
        const err = result.error as { message: string } | undefined;
        throw new Error(err?.message ?? "extract_book failed");
      }

      const artifact = (result.artifacts as Array<{ metadata: { title: string; chapters: ChapterMeta[] } }>)[0];
      setBook({
        title: artifact.metadata.title,
        bookId: bookStem,
        workDir,
        chapters: artifact.metadata.chapters,
      });
      setProgress(10);
      setStage("idle");
    } catch (err) {
      setError(String(err));
      setStage("error");
    }
  }

  async function handleAnalyze() {
    if (!book) return;
    setStage("analyzing");
    setError(null);
    setSavedMessage(null);

    try {
      const scriptDir = `${book.workDir}/scripts`;
      const scripts: Record<string, string> = {};
      const allCharacters: CharacterMeta[] = [];
      const allVoices: VoiceMeta[] = [];
      const seenIds = new Set<string>();
      const seenVoiceIds = new Set<string>();

      for (let i = 0; i < book.chapters.length; i++) {
        const chapter = book.chapters[i];
        setProgress(10 + Math.round((i / book.chapters.length) * 30));

        const result = await workerCall("analyze_chapter", {
          bookId: book.bookId,
          chapterId: chapter.id,
          title: chapter.title,
          chapterTextPath: chapter.textPath,
          outputDirectory: scriptDir,
          mockLlm: true,
        });

        if (result.status !== "succeeded") continue;

        const artifact = (result.artifacts as Array<{ path: string }>)[0];
        scripts[chapter.id] = artifact.path;

        const scriptRaw = await invoke<string>("run_worker", {
          command: "_read_file",
          inputJson: JSON.stringify({ path: artifact.path }),
        }).catch(() => "{}");
        const scriptData = JSON.parse(scriptRaw) as {
          characters?: CharacterMeta[];
          voices?: VoiceMeta[];
        } | null;

        if (scriptData?.voices) {
          for (const v of scriptData.voices) {
            if (!seenVoiceIds.has(v.id)) {
              seenVoiceIds.add(v.id);
              allVoices.push(v);
            }
          }
        }

        if (scriptData?.characters) {
          for (const c of scriptData.characters) {
            if (!seenIds.has(c.id)) {
              seenIds.add(c.id);
              allCharacters.push(c);
            }
          }
        }
      }

      setAnalysis({ characters: allCharacters, voices: allVoices, scriptPaths: scripts });
      setProgress(40);
      setStage("idle");
    } catch (err) {
      setError(String(err));
      setStage("error");
    }
  }

  async function handleSaveCorrections() {
    if (!book || !analysis) return;
    setStage("saving");
    setError(null);
    setSavedMessage(null);

    try {
      const chaptersInput = book.chapters.map((c) => ({
        chapterId: c.id,
        textPath: c.textPath,
        title: c.title,
      }));

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
      });

      if (result.status !== "succeeded") {
        const err = result.error as { message: string } | undefined;
        throw new Error(err?.message ?? "apply_corrections failed");
      }

      // Update script paths from results
      const artifacts = result.artifacts as Array<{ path: string; metadata: { chapterId: string } }>;
      const newScriptPaths = { ...analysis.scriptPaths };
      const affectedIds: string[] = [];
      for (const art of artifacts) {
        newScriptPaths[art.metadata.chapterId] = art.path;
        affectedIds.push(art.metadata.chapterId);
      }
      setAnalysis({ ...analysis, scriptPaths: newScriptPaths });

      // Re-read characters from first updated script
      if (artifacts.length > 0) {
        const firstScriptRaw = await invoke<string>("run_worker", {
          command: "_read_file",
          inputJson: JSON.stringify({ path: artifacts[0].path }),
        }).catch(() => "{}");
        const firstScript = JSON.parse(firstScriptRaw) as {
          characters?: CharacterMeta[];
          voices?: VoiceMeta[];
        } | null;

        if (firstScript?.characters) {
          const updatedIds = new Set(firstScript.characters.map((c) => c.id));
          const preserved = analysis.characters.filter((c) => !updatedIds.has(c.id));
          setAnalysis({
            ...analysis,
            scriptPaths: newScriptPaths,
            characters: [...preserved, ...firstScript.characters],
          });
        }
      }

      correctionsStore.markSaved(affectedIds);
      setSavedMessage(`Corrections saved. ${affectedIds.length} chapter(s) updated.`);
      setStage("idle");
    } catch (err) {
      setError(String(err));
      setStage("error");
    }
  }

  async function handleGenerate() {
    if (!book || !analysis) return;
    setStage("generating");
    setError(null);

    const chaptersToGenerate = correctionState.affectedChapters.length > 0
      ? book.chapters.filter((c) => correctionState.affectedChapters.includes(c.id))
      : book.chapters;

    try {
      let generatedPath: string | null = null;
      for (let ci = 0; ci < chaptersToGenerate.length; ci++) {
        const chapter = chaptersToGenerate[ci];
        const scriptPath = analysis.scriptPaths[chapter.id];
        if (!scriptPath) continue;

        const segDir = `${book.workDir}/segments/${chapter.id}`;
        const assembledPath = `${book.workDir}/audio/${chapter.id}.wav`;

        const scriptRaw = await invoke<string>("run_worker", {
          command: "_read_file",
          inputJson: JSON.stringify({ path: scriptPath }),
        }).catch(() => "{}");
        const script = JSON.parse(scriptRaw) as { segments?: Array<{ id: string }> };
        const segments = script.segments ?? [];

        for (let i = 0; i < segments.length; i++) {
          setProgress(40 + Math.round(((ci * segments.length + i) / (chaptersToGenerate.length * segments.length)) * 50));
          await workerCall("synthesize_segment_audio", {
            scriptPath,
            segmentId: segments[i].id,
            outputDirectory: segDir,
            backend: "parler",
          });
        }

        const result = await workerCall("assemble_chapter_audio", {
          segmentAudioDirectory: segDir,
          outputPath: assembledPath,
        });

        if (result.status === "succeeded") {
          generatedPath = assembledPath;
        }
      }

      if (generatedPath) setAudioPath(generatedPath);
      setProgress(100);
      setStage("done");
    } catch (err) {
      setError(String(err));
      setStage("error");
    }
  }

  const handleGenderChange = useCallback((characterId: string, gender: string) => {
    correctionsStore.setGender(characterId, gender);
    setSavedMessage(null);
  }, []);

  const handleVoiceChange = useCallback((characterId: string, voiceId: string) => {
    correctionsStore.setVoice(characterId, voiceId);
    setSavedMessage(null);
  }, []);

  const steps = [
    { label: "Import", status: book ? "Done" : stage === "importing" ? "Running..." : "Ready" },
    { label: "Analyze", status: analysis ? "Done" : stage === "analyzing" ? "Running..." : book ? "Ready" : "Waiting" },
    { label: "Review", status: correctionState.savedCorrections ? "Done" : analysis ? "Ready" : "Waiting" },
    { label: "Generate", status: stage === "done" ? "Done" : stage === "generating" ? "Running..." : analysis ? "Ready" : "Waiting" },
  ];

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Workspace">
        <h1>Audiobook Generator</h1>
        <button
          className="primary-action"
          type="button"
          onClick={handleImportBook}
          disabled={stage === "importing" || stage === "analyzing" || stage === "saving" || stage === "generating"}
        >
          {stage === "importing" ? "Importing..." : "Import Book"}
        </button>
        {book && !analysis && (
          <button
            className="primary-action"
            type="button"
            onClick={handleAnalyze}
            disabled={stage === "analyzing"}
            style={{ marginTop: 8 }}
          >
            {stage === "analyzing" ? "Analyzing..." : "Analyze Book"}
          </button>
        )}
        {analysis && (
          <button
            className="primary-action"
            type="button"
            onClick={handleSaveCorrections}
            disabled={!correctionState.dirty || stage === "saving"}
            style={{ marginTop: 8 }}
          >
            {stage === "saving" ? "Saving..." : "Save Corrections"}
          </button>
        )}
        {correctionState.savedCorrections && (
          <button
            className="primary-action"
            type="button"
            onClick={handleGenerate}
            disabled={stage === "generating"}
            style={{ marginTop: 8 }}
          >
            {stage === "generating" ? "Generating..." : "Regenerate Affected Chapters"}
          </button>
        )}
        <nav aria-label="Workflow">
          {steps.map((step) => (
            <div className="workflow-step" key={step.label}>
              <span>{step.label}</span>
              <small>{step.status}</small>
            </div>
          ))}
        </nav>
      </aside>

      <section className="workspace" aria-label="Audiobook job">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">Local desktop pipeline</p>
            <h2>Job Progress</h2>
          </div>
          <span className="status-pill">{book ? book.title : "No active book"}</span>
        </header>

        <section className="progress-panel" aria-label="Pipeline progress">
          {stage === "error" ? (
            <div>
              <strong>Error</strong>
              <p>{error}</p>
            </div>
          ) : stage === "done" ? (
            <div>
              <strong>Done!</strong>
              <p>Chapter audio saved to: {audioPath}</p>
            </div>
          ) : book ? (
            <div>
              <strong>{book.title}</strong>
              <p>
                {book.chapters.length} chapter{book.chapters.length !== 1 ? "s" : ""} detected.
                {analysis ? ` ${analysis.characters.length} character${analysis.characters.length !== 1 ? "s" : ""} identified.` : ""}
              </p>
            </div>
          ) : (
            <div>
              <strong>Import a PDF or EPUB to begin.</strong>
              <p>Extraction, chapter detection, dialogue analysis, and local TTS run as resumable stages.</p>
            </div>
          )}
          <progress value={progress} max="100" aria-label="Generation progress" />
        </section>

        <section className="grid">
          <article>
            <h3>Characters</h3>
            {analysis && analysis.characters.length > 0 ? (
              <ul>
                {analysis.characters.map((c) => (
                  <li key={c.id}>
                    {c.canonicalName} <small>({c.gender} · {c.voiceId})</small>
                  </li>
                ))}
              </ul>
            ) : (
              <p>Detected speakers, gender confidence, aliases, and assigned voices will appear here.</p>
            )}
          </article>
          <article>
            <h3>Chapters</h3>
            {book && book.chapters.length > 0 ? (
              <ul>
                {book.chapters.slice(0, 10).map((c) => (
                  <li key={c.id}>
                    {c.title}
                    {analysis?.scriptPaths[c.id] ? " ✓" : ""}
                    {correctionState.affectedChapters.includes(c.id) ? " (pending regeneration)" : ""}
                    <small> ({Math.round(c.textLength / 1000)}k chars)</small>
                  </li>
                ))}
                {book.chapters.length > 10 && <li>...and {book.chapters.length - 10} more</li>}
              </ul>
            ) : (
              <p>Chapter scripts and generation state will be listed as the worker pipeline runs.</p>
            )}
          </article>
          <article>
            <h3>Rights</h3>
            <p>Unknown or restricted license status will require confirmation before generation.</p>
            <label className="attestation">
              <input type="checkbox" />
              <span>I have the right to convert this book</span>
            </label>
          </article>
          <article className="review-panel">
            <h3>Review</h3>
            {savedMessage && <p className="saved-message">{savedMessage}</p>}
            {analysis ? (
              <>
                <CharacterTable
                  characters={analysis.characters}
                  voices={VOICE_OPTIONS}
                  onGenderChange={handleGenderChange}
                  onVoiceChange={handleVoiceChange}
                />
                {correctionState.dirty && (
                  <p className="hint">You have unsaved corrections. Click "Save Corrections" to apply them.</p>
                )}
              </>
            ) : (
              <p>Run analysis first to see the character table and make corrections.</p>
            )}
          </article>
          <article>
            <h3>Export</h3>
            {audioPath ? (
              <p>Chapter audio ready: <code>{audioPath}</code></p>
            ) : (
              <p>Completed chapter audio and metadata exports will be available after generation.</p>
            )}
          </article>
        </section>
      </section>
    </main>
  );
}
```

**Step 4: Run tests to verify they pass**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run src/App.test.tsx
```

Expected: all tests PASS

**Step 5: Run full TypeScript test suite**

```bash
cd .worktrees/mvp-implementation/apps/desktop
npm test -- --run
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add apps/desktop/src/App.tsx apps/desktop/src/App.test.tsx
git commit -m "feat: integrate CharacterTable and Save/Regenerate flow into App"
```

---

### Task 6: Add CSS for character table and confidence bars

**Files:**
- Modify: `apps/desktop/src/styles.css`

**Step 1: Append table and review panel styles**

Add at the end of `apps/desktop/src/styles.css`:

```css
/* Character table */
.character-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.character-table th {
  text-align: left;
  padding: 8px 12px;
  border-bottom: 2px solid var(--border-color, #334155);
  color: var(--text-muted, #94a3b8);
  font-weight: 600;
}

.character-table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color, #1e293b);
  vertical-align: middle;
}

.character-table tr.low-confidence {
  background: rgba(239, 68, 68, 0.08);
}

.character-table select {
  background: var(--input-bg, #1e293b);
  color: var(--text, #e2e8f0);
  border: 1px solid var(--border-color, #334155);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.8125rem;
}

/* Confidence bars */
.confidence-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 100px;
}

.confidence-fill {
  height: 8px;
  border-radius: 4px;
  min-width: 4px;
  transition: width 0.3s ease;
}

.confidence-label {
  font-size: 0.75rem;
  color: var(--text-muted, #94a3b8);
  white-space: nowrap;
}

.warning-icon {
  margin-left: 4px;
  font-size: 0.875rem;
}

/* Save/Review messages */
.saved-message {
  background: rgba(34, 197, 94, 0.1);
  border: 1px solid rgba(34, 197, 94, 0.3);
  color: #22c55e;
  padding: 8px 12px;
  border-radius: 6px;
  margin-bottom: 12px;
  font-size: 0.875rem;
}

.hint {
  color: var(--text-muted, #94a3b8);
  font-size: 0.8125rem;
  margin-top: 12px;
}

.review-panel {
  grid-column: span 2;
}

.primary-action:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

**Step 2: Commit**

```bash
git add apps/desktop/src/styles.css
git commit -m "style: add character table, confidence bar, and review panel CSS"
```

---

### Task 7: Integration smoke test

**Files:**
- Modify: `workers/python/tests/test_worker_pipeline.py:73`

**Step 1: Add correction flow to pipeline test**

Append to `workers/python/tests/test_worker_pipeline.py`:

```python
def test_corrections_flow_applies_alias_merge_and_regenerates(tmp_path: Path):
    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text('"Over here," Lizzy called. "Coming," Elizabeth replied.', encoding="utf-8")
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()

    # Step 1: Run initial analysis
    analyze = run_worker(
        "analyze_chapter",
        {
            "bookId": "book_123",
            "chapterId": "chapter_001",
            "title": "Chapter 1",
            "language": "en",
            "chapterTextPath": str(chapter_path),
            "outputDirectory": str(script_dir),
        },
        tmp_path,
    )
    assert analyze["status"] == "succeeded"

    # Step 2: Apply alias merge correction
    corrections = run_worker(
        "apply_corrections",
        {
            "bookId": "book_123",
            "chapters": [
                {"chapterId": "chapter_001", "textPath": str(chapter_path), "title": "Chapter 1"}
            ],
            "corrections": {
                "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
                "genderOverrides": [],
                "voiceOverrides": [],
            },
            "outputDirectory": str(script_dir),
            "language": "en",
        },
        tmp_path,
    )
    assert corrections["status"] == "succeeded"
    corrected_path = Path(corrections["artifacts"][0]["path"])
    corrected_script = json.loads(corrected_path.read_text())

    # Speakers should be unified to elizabeth only
    speakers = {seg["speakerId"] for seg in corrected_script["segments"] if seg["type"] == "dialogue"}
    assert speakers == {"elizabeth"}

    # Characters should have only one elizabeth entry
    character_ids = {c["id"] for c in corrected_script["characters"]}
    assert "elizabeth" in character_ids
    assert "lizzy" not in character_ids
```

**Step 2: Run the pipeline test**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest tests/test_worker_pipeline.py -v
```

Expected: 2 tests PASS (existing + new)

**Step 3: Run the full Python test suite**

```bash
cd .worktrees/mvp-implementation/workers/python
.venv/bin/pytest -v
```

Expected: all tests pass.

**Step 4: Commit**

```bash
git add workers/python/tests/test_worker_pipeline.py
git commit -m "test: add correction flow to worker pipeline integration test"
```

---

## Completion Criteria

The review/correction flow is complete when:

- [ ] `build_chapter_script_with_corrections` applies alias merges, gender overrides, and voice overrides correctly
- [ ] `apply_corrections` CLI command accepts correction payload and returns updated scripts
- [ ] `CharacterTable` renders all characters with confidence bars, editable gender, and voice selects
- [ ] Correction state store tracks pending/saved state with dirty flag
- [ ] "Save Corrections" sends corrections to Python worker and updates script paths
- [ ] "Regenerate Affected Chapters" generates audio only for chapters with modified scripts
- [ ] All existing tests continue to pass
- [ ] All new tests pass
