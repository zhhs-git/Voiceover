# Review & Correction Flow Design

## Status

Draft.

## Summary

Make the desktop app's review panel fully functional: a per-character editable table for alias merging, gender overrides, and voice reassignment, backed by a Python worker command that re-analyzes chapters with user corrections as hard constraints.

## Components

### 1. Correction State

Track pending and applied corrections in React state:

- `pendingCorrections`: edits not yet saved (dirty state)
- `savedCorrections`: last successfully applied corrections
- `affectedChapters`: set of chapter IDs whose scripts changed after applying corrections
- `dirty`: boolean flag to block regeneration with unsaved changes

### 2. Character Table (Review Panel)

Replace the current hardcoded input fields with a table where each row represents one detected character:

- Canonical name (read-only)
- Aliases (read-only, shown as comma-separated)
- Confidence bar (read-only, colored: green >0.8, yellow >0.5, red <=0.5)
- Gender dropdown (editable, defaults to current gender)
- Voice dropdown (editable, populated from voice registry)
- Multi-select checkboxes for alias merge

Low-confidence characters highlighted with a warning icon.

### 3. Correction Submission

"Save Corrections" button:

1. Serializes all pending corrections into JSON
2. Calls Python worker with new `apply_corrections` command
3. Worker re-runs `build_chapter_script` with corrections as hard constraints:
   - Alias merges resolved before segmentation (replace aliased names in text)
   - Gender overrides applied after LLM analysis (override inferred gender)
   - Voice overrides applied after voice assignment (override assigned voiceId)
4. Returns updated `ChapterScript[]` for affected chapters
5. TypeScript layer updates `analysis.scriptPaths` with new script paths
6. `affectedChapters` set populated, `dirty` flag cleared

### 4. Regeneration Trigger

"Regenerate Affected Chapters" button:

1. Only enabled when `savedCorrections` exist and `!dirty`
2. Iterates over `affectedChapters`, re-runs TTS for each chapter's segments
3. Shows per-chapter progress in the progress bar
4. Updates `audioPath` for each regenerated chapter

## Python Worker Change

### New command: `apply_corrections`

Input:
```json
{
  "bookId": "string",
  "chapters": [
    {"chapterId": "string", "textPath": "string", "title": "string"}
  ],
  "corrections": {
    "aliasMerges": [{"from": "string", "to": "string"}],
    "genderOverrides": [{"characterId": "string", "gender": "female|male|neutral"}],
    "voiceOverrides": [{"characterId": "string", "voiceId": "string"}]
  },
  "outputDirectory": "string",
  "language": "string"
}
```

Output:
```json
{
  "status": "succeeded",
  "warnings": [],
  "artifacts": [
    {
      "kind": "chapter_script",
      "path": "string",
      "metadata": {
        "chapterId": "string",
        "characterCount": 0,
        "segmentCount": 0
      }
    }
  ]
}
```

## Data Flow

```
User edits character table
  → pendingCorrections updated (dirty=true)
  → "Save Corrections" clicked
  → TypeScript calls Python worker: apply_corrections
  → Worker applies alias merge (text substitution), re-analyzes, overrides gender/voice
  → Worker returns updated scripts
  → TypeScript updates analysis.scriptPaths, sets affectedChapters
  → dirty=false, savedCorrections updated
  → "Regenerate Affected Chapters" clicked
  → TypeScript calls Python worker: synthesize_segment_audio + assemble_chapter_audio per affected chapter
  → audioPath updated for regenerated chapters
```

## Error Handling

- Worker failures display inline error with retry option
- Individual segment TTS failures don't block other segments
- Stale script check: if book text changed since last analysis, warn before applying corrections

## Testing

- Unit tests for correction serialization/deserialization
- Unit tests for character table component (renders all characters, editable fields respond)
- Integration test: apply corrections via worker, verify script changed as expected
- Smoke test: full correction → regeneration flow in the desktop app
