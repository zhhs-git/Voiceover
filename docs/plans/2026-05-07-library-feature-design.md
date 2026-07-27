# Library Feature Design

## Status

Draft.

## Summary

Replace the single-book linear pipeline with a library-first architecture. The app opens to a Library view showing all imported books. Clicking a book opens its detail view with inline pipeline steps (analyze, review, generate). Character lists are cumulative per book — analyzing new chapters adds characters without losing previous assignments. Book import is deduplicated by source file path. All generated files move from system temp directories to a persistent app data directory.

## Components

### 1. Top-Level Routing

Replace the current state-machine `App.tsx` with two views controlled by a `currentView` state:

```ts
type View =
  | { page: "library" }
  | { page: "bookDetail"; bookId: string }
```

`LibraryView` and `BookDetailView` are sibling components rendered conditionally. No external router library needed — `useState` is sufficient for two views.

### 2. Data Model Changes

#### books table — add timestamps

```sql
ALTER TABLE books ADD COLUMN imported_at TEXT;
ALTER TABLE books ADD COLUMN updated_at TEXT;
```

#### New characters table — cumulative per book

```sql
CREATE TABLE characters (
  id TEXT PRIMARY KEY,
  book_id TEXT NOT NULL REFERENCES books(id),
  canonical_name TEXT NOT NULL,
  gender TEXT,
  voice_id TEXT,
  confidence REAL DEFAULT 0.0,
  aliases TEXT DEFAULT '[]',  -- JSON array of strings
  updated_at TEXT
);
```

On analysis, characters are upserted (`INSERT OR REPLACE`). Re-analyzing a chapter updates existing characters, and new characters are added. The cumulative list grows across analysis sessions.

### 3. Library View

Home screen. Displays all imported books in a card grid.

Each card shows:
- Title (truncated if long)
- Progress bar (chapters generated / total)
- Import date
- Click → navigates to `BookDetailView`

Import button (`+ Import`) in the top-right corner opens a native file dialog (`.epub`, `.pdf`).

Empty state: "No books yet. Import your first book to get started." with centered import button.

Data source: `SELECT id, title, source_path, imported_at FROM books ORDER BY imported_at DESC`. Chapter counts queried from `chapters` table per book.

### 4. Import Flow with Dedup

1. User clicks `+ Import` → native file dialog opens
2. On file selection, check `books` table for existing row with same `source_path`
3. If found: show toast "Already imported" and navigate to that book's detail view
4. If not found: run extraction (same worker flow as today) with work_dir set to persistent path:
   ```
   ~/.config/audiobook-generator/books/{bookId}/
   ```
5. Persist book in SQLite, save extraction cache
6. Auto-navigate to the new book's detail view

### 5. Persistent Storage

All new books use persistent work directories instead of system temp dirs:

```
~/.config/audiobook-generator/books/{bookId}/
├── chapters/
│   └── {chapterId}.txt
├── scripts/
│   └── {chapterId}.json
├── segments/
│   └── {chapterId}/
│       └── {segmentId}.wav
├── audio/
│   └── {chapterId}.wav
└── book-extraction.json
```

Same subdirectory structure as today, just rooted in app data instead of `Tauri tempDir`.

### 6. Book Detail View

Split-panel layout: chapter list on the left, tabbed action panels on the right.

#### Left Panel — Chapter List

- Checkboxes next to chapters not yet analyzed/generated
- Status icons: ✓ (analyzed), 🔊 (generated), (—) pending
- "Select All" checkbox
- Action button (label changes based on selected tab: "Analyze Selected", "Generate Selected")

#### Right Panel — Tabbed Actions

**Analyze tab:**
- Runs LLM analysis on selected chapters
- Shows per-chapter progress
- On completion: upserts characters into `characters` table, saves scripts

**Review tab:**
- Cumulative character table (from SQLite `characters` table)
- Same editing as today: gender dropdown, voice dropdown, voice preview
- "Save Corrections" applies to all chapters (worker `apply_corrections`)

**Generate tab:**
- Chapter list with status badges and per-chapter "Generate" button
- Progress bar with ETA during generation
- Audio player (`<audio>` element) for each generated chapter
- "Regenerate All" button in header

#### Bottom Strip

Compact character summary strip across the bottom — quick glance at current character list. Click jumps to Review tab.

#### Back Navigation

Back arrow (← Library) in the header returns to Library view.

### 7. Cumulative Characters

Characters are stored per-book in the `characters` table. On each analysis run:
1. Existing characters for the book are loaded from SQLite
2. Newly detected characters are merged (upsert by `id`)
3. Conflicts resolved: new analysis data wins (higher confidence or more recent)
4. User corrections (gender/voice overrides) are preserved across re-analysis

This ensures TTS voices remain consistent across all chapters in a book.

## Data Flow

```
Library View
  ├── Import: file dialog → dedup check → extract → persist → BookDetail
  └── Click book: → BookDetail View

Book Detail View
  ├── Select chapters → Analyze tab: LLM worker → save scripts + upsert characters
  ├── Review tab: load characters → edit → apply corrections worker
  ├── Generate tab: TTS worker → save audio → play
  └── Back arrow: → Library View
```

## Changes Needed

| Area | Change |
|------|--------|
| `apps/desktop/src/App.tsx` | Replace state machine with view routing |
| `apps/desktop/src/components/LibraryView.tsx` | New: book grid + import button |
| `apps/desktop/src/components/BookDetailView.tsx` | New: split-panel with tabs |
| `apps/desktop/src/state/store.ts` | Add `listBooks`, queries for characters, upsert helpers |
| `apps/desktop/src/components/containers/` | Extract pipeline logic from containers for reuse in tabs |
| `apps/desktop/src-tauri/src/lib.rs` | Add `characters` table migration, `db_list_books`, `db_upsert_character`, `db_get_characters`, dedup query |
| `apps/desktop/src/hooks/` | Refactor hooks to work outside linear pipeline context |
| `apps/desktop/src/components/Sidebar.tsx` | Replaced by book detail left panel |
| `apps/desktop/src/types.ts` | Add `LibraryBook`, `BookCharacter` types |

## Non-Goals

- Tagging, collections, or search in library
- Multi-language support
- M4B export
- Migrating existing temp-dir books (they appear in library but files may be gone)
- Cross-device sync
