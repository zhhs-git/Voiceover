# Library Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-book linear pipeline with a library-first architecture — library home screen, per-book detail with inline pipeline, cumulative characters, deduplicated imports, persistent storage.

**Architecture:** Two-view routing in App.tsx (`library` / `bookDetail`). Library shows book grid with import button. Book detail is split-panel: chapter list (left) + tabbed actions (right: analyze/review/generate). Characters table in SQLite grows cumulatively per book. All generated files stored in `~/.config/audiobook-generator/books/{bookId}/` instead of temp dirs.

**Tech Stack:** Tauri (Rust + rusqlite), React 19 + TypeScript 5.6, Vite 6, Vitest

---

### Task 1: Rust Backend — Schema Migration & New Commands

**Files:**
- Modify: `apps/desktop/src-tauri/src/lib.rs` (entire file)
- Modify: `apps/desktop/src-tauri/Cargo.toml` (check deps)

**Step 1: Add `imported_at`/`updated_at` columns to books table and create `characters` table**

Modify the `CREATE TABLE IF NOT EXISTS` batch in `get_db()`:

```rust
conn.execute_batch(
    "CREATE TABLE IF NOT EXISTS books (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
        source_language TEXT NOT NULL, output_language TEXT NOT NULL, work_dir TEXT NOT NULL,
        imported_at TEXT, updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS chapters (
        id TEXT NOT NULL, book_id TEXT NOT NULL, title TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', script_path TEXT,
        PRIMARY KEY (id, book_id)
    );
    CREATE TABLE IF NOT EXISTS characters (
        id TEXT NOT NULL, book_id TEXT NOT NULL, canonical_name TEXT NOT NULL,
        gender TEXT, voice_id TEXT, confidence REAL DEFAULT 0.0,
        aliases TEXT DEFAULT '[]', updated_at TEXT,
        PRIMARY KEY (id, book_id)
    );"
)
```

Also add a migration for existing databases — after `execute_batch`, run:

```rust
// Add columns if they don't exist (safe migration for existing DBs)
for col in &["imported_at", "updated_at"] {
    let _ = conn.execute(
        &format!("ALTER TABLE books ADD COLUMN {} TEXT", col),
        [],
    );
}
```

**Step 2: Add Rust commands**

Below the existing `db_get_chapters_with_scripts` fn, add:

```rust
#[tauri::command]
fn db_list_books() -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, title, source_path, work_dir, imported_at FROM books ORDER BY imported_at DESC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "title": row.get::<_, String>(1)?,
                "sourcePath": row.get::<_, String>(2)?,
                "workDir": row.get::<_, String>(3)?,
                "importedAt": row.get::<_, Option<String>>(4)?
            }))
        })
        .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row.map_err(|e| e.to_string())?);
    }
    Ok(result)
}

#[tauri::command]
fn db_get_book(source_path: String) -> Result<Option<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, title, source_path, work_dir, imported_at FROM books WHERE source_path = ?1")
        .map_err(|e| e.to_string())?;
    let mut rows = stmt
        .query_map(rusqlite::params![source_path], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "title": row.get::<_, String>(1)?,
                "sourcePath": row.get::<_, String>(2)?,
                "workDir": row.get::<_, String>(3)?,
                "importedAt": row.get::<_, Option<String>>(4)?
            }))
        })
        .map_err(|e| e.to_string())?;
    if let Some(row) = rows.next() {
        Ok(Some(row.map_err(|e| e.to_string())?))
    } else {
        Ok(None)
    }
}

#[tauri::command]
fn db_upsert_character(
    id: String, book_id: String, canonical_name: String,
    gender: Option<String>, voice_id: Option<String>,
    confidence: Option<f64>, aliases: Option<String>,
) -> Result<(), String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let now = chrono::Utc::now().to_rfc3339();
    db.execute(
        "INSERT OR REPLACE INTO characters (id, book_id, canonical_name, gender, voice_id, confidence, aliases, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![id, book_id, canonical_name, gender, voice_id, confidence, aliases, now],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn db_get_characters(book_id: String) -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, canonical_name, gender, voice_id, confidence, aliases FROM characters WHERE book_id = ?1")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![book_id], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "canonicalName": row.get::<_, String>(1)?,
                "gender": row.get::<_, Option<String>>(2)?,
                "voiceId": row.get::<_, Option<String>>(3)?,
                "confidence": row.get::<_, f64>(4)?,
                "aliases": row.get::<_, String>(5)?
            }))
        })
        .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row.map_err(|e| e.to_string())?);
    }
    Ok(result)
}

#[tauri::command]
fn db_get_chapters(book_id: String) -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, title, status, script_path FROM chapters WHERE book_id = ?1")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![book_id], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "title": row.get::<_, String>(1)?,
                "status": row.get::<_, String>(2)?,
                "scriptPath": row.get::<_, Option<String>>(3)?
            }))
        })
        .map_err(|e| e.to_string())?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row.map_err(|e| e.to_string())?);
    }
    Ok(result)
}
```

**Step 3: Update `db_create_book` to set `imported_at`**

```rust
#[tauri::command]
fn db_create_book(id: String, title: String, source_path: String, work_dir: String) -> Result<(), String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let now = chrono::Utc::now().to_rfc3339();
    db.execute(
        "INSERT OR REPLACE INTO books (id, title, source_path, source_language, output_language, work_dir, imported_at, updated_at) VALUES (?1, ?2, ?3, 'en', 'en', ?4, ?5, ?5)",
        rusqlite::params![id, title, source_path, work_dir, now],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
```

**Step 4: Add `chrono` dependency to Cargo.toml**

Under `[dependencies]`:

```toml
chrono = { version = "0.4", features = ["serde"] }
```

**Step 5: Register new commands in `run()`**

```rust
.invoke_handler(tauri::generate_handler![
    run_worker, copy_file,
    db_create_book, db_upsert_chapter, db_get_chapters_with_scripts,
    db_list_books, db_get_book, db_upsert_character, db_get_characters, db_get_chapters,
])
```

**Step 6: Build Rust backend to verify compilation**

Run: `cd apps/desktop/src-tauri && cargo check`
Expected: Compilation succeeds with no errors.

**Step 7: Commit**

```bash
git add apps/desktop/src-tauri/
git commit -m "feat: add characters table, list/query commands, timestamp columns"
```

---

### Task 2: Frontend Types & Store — New API Surface

**Files:**
- Modify: `apps/desktop/src/types.ts`
- Modify: `apps/desktop/src/state/store.ts`

**Step 1: Add new types to `types.ts`**

Append after the existing types:

```ts
export interface LibraryBook {
  id: string;
  title: string;
  sourcePath: string;
  workDir: string;
  importedAt: string | null;
}

export interface ChapterRecord {
  id: string;
  title: string;
  status: string;
  scriptPath: string | null;
}

export interface CharacterRecord {
  id: string;
  canonicalName: string;
  gender: string | null;
  voiceId: string | null;
  confidence: number;
  aliases: string; // JSON array
}

export type AppView =
  | { page: "library" }
  | { page: "bookDetail"; bookId: string };
```

**Step 2: Extend store in `store.ts`**

Add new methods to `createAudiobookStore()`:

```ts
export function createAudiobookStore() {
  return {
    createBook(record: { id: string; title: string; sourcePath: string; workDir: string }) {
      return invoke("db_create_book", { id: record.id, title: record.title, sourcePath: record.sourcePath, workDir: record.workDir });
    },
    upsertChapter(record: { id: string; bookId: string; title: string; status: string; scriptPath?: string }) {
      return invoke("db_upsert_chapter", { id: record.id, bookId: record.bookId, title: record.title, status: record.status, scriptPath: record.scriptPath ?? null });
    },
    async getChaptersWithScripts(bookId: string): Promise<Array<{ id: string; scriptPath: string }>> {
      return await invoke("db_get_chapters_with_scripts", { bookId }) as any;
    },
    async listBooks(): Promise<LibraryBook[]> {
      return await invoke("db_list_books") as any;
    },
    async getBook(sourcePath: string): Promise<LibraryBook | null> {
      return await invoke("db_get_book", { sourcePath }) as any;
    },
    async upsertCharacter(record: {
      id: string; bookId: string; canonicalName: string;
      gender?: string | null; voiceId?: string | null;
      confidence?: number; aliases?: string;
    }) {
      return invoke("db_upsert_character", {
        id: record.id, bookId: record.bookId,
        canonicalName: record.canonicalName,
        gender: record.gender ?? null,
        voiceId: record.voiceId ?? null,
        confidence: record.confidence ?? 0.0,
        aliases: record.aliases ?? "[]",
      });
    },
    async getCharacters(bookId: string): Promise<CharacterRecord[]> {
      return await invoke("db_get_characters", { bookId }) as any;
    },
    async getChapters(bookId: string): Promise<ChapterRecord[]> {
      return await invoke("db_get_chapters", { bookId }) as any;
    },
  };
}
```

You'll need to add the import for `LibraryBook` and `CharacterRecord` types at the top.

**Step 3: Typecheck**

Run: `cd apps/desktop && npx tsc --noEmit`
Expected: No errors.

**Step 4: Commit**

```bash
git add apps/desktop/src/types.ts apps/desktop/src/state/store.ts
git commit -m "feat: add library types and store API for list/query/characters"
```

---

### Task 3: Library View Component

**Files:**
- Create: `apps/desktop/src/components/LibraryView.tsx`
- Create: `apps/desktop/src/components/LibraryView.test.tsx` (optional, skip if no existing test patterns)

**Step 1: Write LibraryView component**

```tsx
import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import type { LibraryBook } from "../types";
import { createAudiobookStore } from "../state/store";

const db = createAudiobookStore();

interface LibraryViewProps {
  onImport: () => void;
  onSelectBook: (book: LibraryBook) => void;
  importError: string | null;
}

function chapterProgressText(book: LibraryBook, chapters: Map<string, { total: number; generated: number }>): string {
  const info = chapters.get(book.id);
  if (!info) return "—";
  if (info.generated === 0) return `${info.total} chapters`;
  return `${info.generated}/${info.total} generated`;
}

function progressPercent(book: LibraryBook, chapters: Map<string, { total: number; generated: number }>): number {
  const info = chapters.get(book.id);
  if (!info || info.total === 0) return 0;
  return Math.round((info.generated / info.total) * 100);
}

export function LibraryView({ onImport, onSelectBook, importError }: LibraryViewProps) {
  const [books, setBooks] = useState<LibraryBook[]>([]);
  const [chapterInfo, setChapterInfo] = useState<Map<string, { total: number; generated: number }>>(new Map());

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const list = await db.listBooks();
      if (cancelled) return;
      setBooks(list);

      const info = new Map<string, { total: number; generated: number }>();
      for (const b of list) {
        const chapters = await db.getChapters(b.id);
        const generated = chapters.filter(c => c.status === "succeeded").length;
        info.set(b.id, { total: chapters.length, generated });
      }
      if (!cancelled) setChapterInfo(info);
    }
    load();
    return () => { cancelled = true; };
  }, []);

  if (books.length === 0) {
    return (
      <main className="library-view">
        <div className="library-empty">
          <h2>No books yet</h2>
          <p>Import your first book to get started.</p>
          <button className="btn-primary" onClick={onImport}>
            + Import Book
          </button>
          {importError && <p className="error-text">{importError}</p>}
        </div>
      </main>
    );
  }

  return (
    <main className="library-view">
      <header className="library-header">
        <h1>Library</h1>
        <button className="btn-primary" onClick={onImport}>
          + Import
        </button>
      </header>
      {importError && <p className="error-text">{importError}</p>}
      <div className="library-grid">
        {books.map((book) => {
          const pct = progressPercent(book, chapterInfo);
          return (
            <button
              key={book.id}
              className="library-card"
              onClick={() => onSelectBook(book)}
            >
              <div className="card-cover">📖</div>
              <div className="card-title">{book.title}</div>
              <div className="card-progress">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="progress-text">
                  {chapterProgressText(book, chapterInfo)}
                </span>
              </div>
              <div className="card-date">{book.importedAt?.split("T")[0] ?? ""}</div>
            </button>
          );
        })}
      </div>
    </main>
  );
}
```

**Step 2: Commit**

```bash
git add apps/desktop/src/components/LibraryView.tsx
git commit -m "feat: add library view component"
```

---

### Task 4: Book Detail View Component

**Files:**
- Create: `apps/desktop/src/components/BookDetailView.tsx`

This component integrates all pipeline steps (analyze, review, generate) into tabs within a split-panel layout.

**Step 1: Write BookDetailView**

```tsx
import { useRef, useState, useSyncExternalStore, useCallback } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import type { AnalysisState, BookState, ChapterMeta, CharacterMeta, LibraryBook, PipelineStage, ProgressDetail, VoiceMeta, VoiceOption } from "../types";
import { createAudiobookStore } from "../state/store";
import { createCorrectionsStore, type CorrectionState } from "../state/corrections";
import { useChapterAnalysis } from "../hooks/useChapterAnalysis";
import { useGeneration } from "../hooks/useGeneration";
import { workerCall } from "../lib/workerCall";

const db = createAudiobookStore();

const VOICE_OPTIONS: VoiceOption[] = [
  { id: "narrator_default", displayName: "Default Narrator" },
  { id: "female_adult_01", displayName: "Female Adult 01" },
  { id: "male_adult_01", displayName: "Male Adult 01" },
  { id: "neutral_dialogue_01", displayName: "Neutral Dialogue 01" },
];

type DetailTab = "analyze" | "review" | "generate";

interface BookDetailViewProps {
  libraryBook: LibraryBook;
  book: BookState;
  onBack: () => void;
}

export function BookDetailView({ libraryBook, book, onBack }: BookDetailViewProps) {
  const [tab, setTab] = useState<DetailTab>("analyze");
  const [analysis, setAnalysis] = useState<AnalysisState | null>(null);
  const [chapterAudioPaths, setChapterAudioPaths] = useState<Record<string, string>>({});
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [analyzeProgress, setAnalyzeProgress] = useState("");
  const [chapterStatuses, setChapterStatuses] = useState<Record<string, string>>({});
  const [progressDetail, setProgressDetail] = useState<ProgressDetail[]>([]);
  const [selectedChapters, setSelectedChapters] = useState<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const [showCharacterPreview, setShowCharacterPreview] = useState(false);

  const correctionsStoreRef = useRef(createCorrectionsStore());
  const correctionState = useSyncExternalStore(
    correctionsStoreRef.current.subscribe,
    correctionsStoreRef.current.get,
  );

  const isBusy = stage === "importing" || stage === "analyzing" || stage === "saving" || stage === "generating";

  const { handleAnalyze } = useChapterAnalysis({
    book, selectedChapters, setStage, setError, setSavedMessage,
    setAnalyzeProgress, setChapterStatuses, setProgressDetail, setProgress,
    setAnalysis, setCurrentStep: () => {}, abortRef, db,
  });

  const { handleGenerate, handleRegenerateChapter, handleRegenerateAll } = useGeneration({
    book, analysis, selectedChapters, chapterAudioPaths,
    correctionState: correctionState as { affectedChapters: string[]; dirty?: boolean },
    setStage, setError, setAnalyzeProgress, setProgressDetail, setProgress,
    setChapterAudioPaths, setCurrentStep: () => {}, abortRef,
  });

  function toggleChapter(chapterId: string) {
    setSelectedChapters(prev => {
      const next = new Set(prev);
      next.has(chapterId) ? next.delete(chapterId) : next.add(chapterId);
      return next;
    });
  }

  function toggleAllChapters() {
    const allSelected = book.chapters.length > 0 && book.chapters.every(c => selectedChapters.has(c.id));
    setSelectedChapters(allSelected ? new Set() : new Set(book.chapters.map(c => c.id)));
  }

  function handleStop() { abortRef.current?.abort(); setAnalyzeProgress("Stopping..."); }

  const handleGenderChange = useCallback((characterId: string, gender: string) => {
    correctionsStoreRef.current.setGender(characterId, gender);
    setSavedMessage(null);
  }, []);

  const handleVoiceChange = useCallback((characterId: string, voiceId: string) => {
    correctionsStoreRef.current.setVoice(characterId, voiceId);
    setAnalysis(current => {
      if (!current) return current;
      return {
        ...current,
        characters: current.characters.map(c =>
          c.id === characterId ? { ...c, voiceId } : c
        ),
      };
    });
    setSavedMessage(null);
  }, []);

  async function handleSaveCorrections() {
    if (!book || !analysis) return;
    setStage("saving");
    setError(null);
    setSavedMessage(null);
    try {
      const result = await workerCall("apply_corrections", {
        bookId: book.bookId,
        chapters: book.chapters.map(c => ({ chapterId: c.id, textPath: c.textPath, title: c.title })),
        corrections: {
          aliasMerges: correctionState.aliasMerges,
          genderOverrides: correctionState.genderOverrides,
          voiceOverrides: correctionState.voiceOverrides,
        },
        outputDirectory: `${book.workDir}/scripts`,
        language: "en",
      });
      if (result.status !== "succeeded") throw new Error((result.error as any)?.message ?? "apply_corrections failed");
      const artifacts = result.artifacts as Array<{ path: string; metadata: { chapterId: string } }>;
      const newScriptPaths = { ...analysis.scriptPaths };
      const affectedIds: string[] = [];
      for (const art of artifacts) {
        newScriptPaths[art.metadata.chapterId] = art.path;
        affectedIds.push(art.metadata.chapterId);
      }
      if (artifacts.length > 0) {
        const firstScriptRaw = await invoke<string>("run_worker", { command: "_read_file", inputJson: JSON.stringify({ path: artifacts[0].path }) }).catch(() => "{}");
        const firstScript = JSON.parse(firstScriptRaw) as { characters?: CharacterMeta[]; voices?: VoiceMeta[] } | null;
        if (firstScript?.characters) {
          const updatedIds = new Set(firstScript.characters.map(c => c.id));
          setAnalysis({ ...analysis, scriptPaths: newScriptPaths, characters: [...analysis.characters.filter(c => !updatedIds.has(c.id)), ...firstScript.characters] });
        } else {
          setAnalysis({ ...analysis, scriptPaths: newScriptPaths });
        }
      }
      correctionsStoreRef.current.markSaved(affectedIds);
      setSavedMessage(`${affectedIds.length} chapter(s) updated.`);
      setStage("idle");
    } catch (err) { setError(String(err)); setStage("error"); }
  }

  async function handlePreviewVoice(voiceId: string) {
    if (!book) return;
    try {
      setSavedMessage(`Generating ${voiceId} preview...`);
      const previewDir = `${book.workDir}/voice-previews`;
      const scriptPath = `${previewDir}/${voiceId}.json`;
      await workerCall("_write_file", { path: scriptPath, content: JSON.stringify({ bookId: book.bookId, chapterId: "voice_preview", segments: [{ id: `preview_${voiceId}`, text: "This is a voice preview.", voiceId, emotion: "neutral", intensity: 0.2, pace: "normal" }] }) });
      const result = await workerCall("synthesize_segment_audio", { scriptPath, segmentId: `preview_${voiceId}`, outputDirectory: previewDir, backend: "kokoro" });
      if (result.status !== "succeeded") throw new Error((result.error as any)?.message ?? "voice preview failed");
      await new Audio(convertFileSrc((result.artifacts as Array<{ path: string }>)[0].path)).play();
      setSavedMessage(`Playing ${voiceId} preview.`);
    } catch (err) { setError(String(err)); setStage("error"); }
  }

  function chapterStatusIcon(chapter: ChapterMeta): string {
    if (chapterAudioPaths[chapter.id]) return "✅";
    if (analysis?.scriptPaths[chapter.id]) return "✓";
    return "—";
  }

  return (
    <main className="book-detail">
      <header className="detail-header">
        <button className="btn-back" onClick={onBack}>← Library</button>
        <h1>{book.title}</h1>
        <button className="btn-secondary" onClick={handleRegenerateAll}>Regen All</button>
      </header>

      <div className="detail-body">
        <aside className="chapter-list">
          <h3>Chapters</h3>
          <label className="select-all">
            <input type="checkbox" checked={book.chapters.length > 0 && book.chapters.every(c => selectedChapters.has(c.id))} onChange={toggleAllChapters} />
            Select All
          </label>
          {book.chapters.map(ch => (
            <label key={ch.id} className="chapter-item">
              <input type="checkbox" checked={selectedChapters.has(ch.id)} onChange={() => toggleChapter(ch.id)} />
              <span className="chapter-status">{chapterStatusIcon(ch)}</span>
              <span className="chapter-title">{ch.title}</span>
            </label>
          ))}
          <button className="btn-primary" onClick={() => { if (tab === "analyze") handleAnalyze(); else handleGenerate(); }}>
            {tab === "analyze" ? "Analyze Selected" : tab === "review" ? "Save Corrections" : "Generate Selected"}
          </button>
        </aside>

        <section className="detail-content">
          <nav className="detail-tabs">
            <button className={`tab-btn ${tab === "analyze" ? "active" : ""}`} onClick={() => setTab("analyze")}>Analyze</button>
            <button className={`tab-btn ${tab === "review" ? "active" : ""}`} onClick={() => setTab("review")}>Review</button>
            <button className={`tab-btn ${tab === "generate" ? "active" : ""}`} onClick={() => setTab("generate")}>Generate</button>
          </nav>

          {tab === "analyze" && (
            <div className="tab-panel">
              {isBusy && <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>}
              <p>{analyzeProgress}</p>
              {Object.entries(chapterStatuses).map(([id, status]) => (
                <div key={id} className="chapter-status-row">{id}: {status}</div>
              ))}
            </div>
          )}

          {tab === "review" && analysis && (
            <div className="tab-panel">
              <table className="character-table">
                <thead><tr><th>Character</th><th>Gender</th><th>Voice</th><th>Preview</th></tr></thead>
                <tbody>
                  {analysis.characters.map(c => (
                    <tr key={c.id}>
                      <td>{c.canonicalName}</td>
                      <td>
                        <select value={c.gender} onChange={e => handleGenderChange(c.id, e.target.value)}>
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                          <option value="neutral">Neutral</option>
                        </select>
                      </td>
                      <td>
                        <select value={c.voiceId} onChange={e => handleVoiceChange(c.id, e.target.value)}>
                          {VOICE_OPTIONS.map(v => <option key={v.id} value={v.id}>{v.displayName}</option>)}
                        </select>
                      </td>
                      <td><button onClick={() => handlePreviewVoice(c.voiceId)}>▶</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button className="btn-primary" onClick={handleSaveCorrections} disabled={!correctionState.dirty}>Save Corrections</button>
              {savedMessage && <p className="success-text">{savedMessage}</p>}
            </div>
          )}

          {tab === "generate" && (
            <div className="tab-panel">
              {isBusy && <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>}
              <p>{analyzeProgress}</p>
              {progressDetail.map(d => <div key={d.label}>{d.label}: {d.value}</div>)}
              {book.chapters.filter(ch => chapterAudioPaths[ch.id]).map(ch => (
                <div key={ch.id} className="audio-row">
                  <span>{ch.title}</span>
                  <audio controls src={convertFileSrc(chapterAudioPaths[ch.id])} />
                  <button onClick={() => handleRegenerateChapter(ch)}>Regen</button>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {error && <div className="error-banner">{error}<button onClick={() => { setError(null); setStage("idle"); }}>Dismiss</button></div>}

      {book.chapters.length > 0 && (
        <footer className="character-strip" onClick={() => { setTab("review"); }}>
          Characters: {analysis ? `${analysis.characters.length} detected` : "None analyzed yet"}
          {analysis && analysis.characters.slice(0, 5).map(c => (
            <span key={c.id} className="character-chip">{c.canonicalName}</span>
          ))}
          {analysis && analysis.characters.length > 5 && <span>+{analysis.characters.length - 5} more</span>}
        </footer>
      )}
    </main>
  );
}
```

**Step 2: Typecheck**

Run: `cd apps/desktop && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add apps/desktop/src/components/BookDetailView.tsx
git commit -m "feat: add book detail view with tabbed pipeline"
```

---

### Task 5: Refactor App.tsx — Two-View Routing

**Files:**
- Modify: `apps/desktop/src/App.tsx` (entire file)
- Modify: `apps/desktop/src/App.test.tsx` (update to match new structure)
- Delete (later): `apps/desktop/src/components/containers/` (5 files) — addressed in Task 7

**Step 1: Rewrite App.tsx**

Replace the entire App.tsx with library-first routing:

```tsx
import { useCallback, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { tempDir } from "@tauri-apps/api/path";
import { invoke } from "@tauri-apps/api/core";
import type { AppView, BookState, LibraryBook } from "./types";
import { LibraryView } from "./components/LibraryView";
import { BookDetailView } from "./components/BookDetailView";
import { createAudiobookStore } from "./state/store";
import { workerCall } from "./lib/workerCall";
import { cachedBookFromExtraction, extractionCachePath, writeExtractionCache } from "./lib/importCache";

const db = createAudiobookStore();

function getBookStem(path: string): string {
  return path.split("/").pop()?.replace(/\.[^.]+$/, "") ?? "book";
}

export function App() {
  const [view, setView] = useState<AppView>({ page: "library" });
  const [activeBook, setActiveBook] = useState<BookState | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const navigateToLibrary = useCallback(() => {
    setView({ page: "library" });
    setImportError(null);
  }, []);

  const navigateToBook = useCallback((book: BookState) => {
    setActiveBook(book);
    setView({ page: "bookDetail", bookId: book.bookId });
  }, []);

  const handleImport = useCallback(async () => {
    const path = await open({
      multiple: false,
      filters: [{ name: "Book", extensions: ["epub", "pdf"] }],
    });
    if (!path) return;

    const sourcePath = path as string;

    // Dedup check
    try {
      const existing = await db.getBook(sourcePath);
      if (existing) {
        setImportError(`"${existing.title}" is already in your library. Opening it now.`);
        // Load existing book from workDir
        const cache = await cachedBookFromExtraction({
          cachePath: extractionCachePath(existing.workDir),
          sourcePath,
          readJson: async (p) => await invoke("run_worker", { command: "_read_file", inputJson: JSON.stringify({ path: p }) }),
        });
        if (cache) {
          navigateToBook(cache);
          return;
        }
        // Fallback: create minimal BookState
        const chapters = await db.getChapters(existing.id);
        navigateToBook({
          title: existing.title,
          bookId: existing.id,
          workDir: existing.workDir,
          chapters: chapters.map(c => ({ id: c.id, title: c.title, textLength: 0, textPath: `${existing.workDir}/chapters/${c.id}.txt` })),
        });
        return;
      }
    } catch (e) {
      // Non-critical, proceed with import
    }

    // New import flow
    setImportError(null);
    try {
      const tmp = await tempDir();
      const bookStem = getBookStem(sourcePath);
      const bookId = `${bookStem}_${Date.now()}`;
      const workDir = `${tmp}/audiobook-generator/${bookStem}`;

      let extracted = await cachedBookFromExtraction({
        cachePath: extractionCachePath(workDir),
        sourcePath,
        readJson: async (p) => await invoke("run_worker", { command: "_read_file", inputJson: JSON.stringify({ path: p }) }),
      });

      if (!extracted) {
        const result = await workerCall("extract_book", {
          bookPath: sourcePath,
          outputDirectory: `${workDir}/chapters`,
        });
        if (result.status !== "succeeded") {
          throw new Error((result.error as any)?.message ?? "extract_book failed");
        }
        const artifact = (result.artifacts as Array<{ metadata: { title: string; chapters: { id: string; title: string; textLength: number; textPath: string }[] } }>)[0];
        extracted = {
          title: artifact.metadata.title,
          bookId,
          workDir,
          chapters: artifact.metadata.chapters,
        };
        await writeExtractionCache({
          sourcePath,
          book: extracted,
          writeJson: async (p, payload) => {
            await workerCall("_write_file", { path: p, content: JSON.stringify(payload) });
          },
        });
      }

      await db.createBook({
        id: bookId,
        title: extracted.title,
        sourcePath,
        workDir: extracted.workDir,
      });

      navigateToBook(extracted);
    } catch (err) {
      setImportError(String(err));
    }
  }, [navigateToBook]);

  if (view.page === "library") {
    return (
      <LibraryView
        onImport={handleImport}
        onSelectBook={async (libraryBook: LibraryBook) => {
          const cache = await cachedBookFromExtraction({
            cachePath: extractionCachePath(libraryBook.workDir),
            sourcePath: libraryBook.sourcePath,
            readJson: async (p) => await invoke("run_worker", { command: "_read_file", inputJson: JSON.stringify({ path: p }) }),
          });
          if (cache) {
            navigateToBook(cache);
            return;
          }
          const chapters = await db.getChapters(libraryBook.id);
          navigateToBook({
            title: libraryBook.title,
            bookId: libraryBook.id,
            workDir: libraryBook.workDir,
            chapters: chapters.map(c => ({ id: c.id, title: c.title, textLength: 0, textPath: `${libraryBook.workDir}/chapters/${c.id}.txt` })),
          });
        }}
        importError={importError}
      />
    );
  }

  if (activeBook && view.page === "bookDetail") {
    const libBook: LibraryBook = {
      id: activeBook.bookId,
      title: activeBook.title,
      sourcePath: "",
      workDir: activeBook.workDir,
      importedAt: null,
    };
    return <BookDetailView libraryBook={libBook} book={activeBook} onBack={navigateToLibrary} />;
  }

  return null;
}
```

**Step 2: Run tests & typecheck**

Run: `cd apps/desktop && npx tsc --noEmit && npm test -- --run`
Expected: Typecheck passes. Tests may need updating (the old App test references old pipeline — we'll handle that in a later task).

**Step 3: Commit**

```bash
git add apps/desktop/src/App.tsx
git commit -m "refactor: replace single-book pipeline with two-view library routing"
```

---

### Task 6: Refactor useChapterAnalysis to Persist Characters

**Files:**
- Modify: `apps/desktop/src/hooks/useChapterAnalysis.ts`

**Step 1: Add character persistence**

After the line `for (const c of scriptData?.characters ?? [])` (around line 156), add a call to `db.upsertCharacter`. Add `db` with `upsertCharacter` to the dependencies interface and the hook:

In the dependencies interface (`UseChapterAnalysisDeps`), change the `db` type from:
```ts
db: { upsertChapter: ... }
```
to:
```ts
db: {
  upsertChapter: (record: { ... }) => Promise<unknown>;
  upsertCharacter: (record: { id: string; bookId: string; canonicalName: string; gender?: string | null; voiceId?: string | null; confidence?: number; aliases?: string; }) => Promise<unknown>;
};
```

Inside the loop, after the character is added to `allCharacters`, add:

```ts
// Persist character to DB
db.upsertCharacter({
  id: c.id,
  bookId: book.bookId,
  canonicalName: c.canonicalName,
  gender: c.gender,
  voiceId: c.voiceId,
  confidence: c.confidence,
  aliases: JSON.stringify(c.aliases),
}).catch(() => {});
```

**Step 2: Typecheck**

Run: `cd apps/desktop && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add apps/desktop/src/hooks/useChapterAnalysis.ts
git commit -m "feat: persist characters to DB on analysis"
```

---

### Task 7: Clean Up Old Pipeline Code

**Files to remove:**
- `apps/desktop/src/components/Sidebar.tsx`
- `apps/desktop/src/components/containers/ImportContainer.tsx`
- `apps/desktop/src/components/containers/AnalyzeContainer.tsx`
- `apps/desktop/src/components/containers/ReviewContainer.tsx`
- `apps/desktop/src/components/containers/GenerateContainer.tsx`
- `apps/desktop/src/components/containers/DoneContainer.tsx`

**Files to check/update for any remaining imports:**
- Any test files referencing these components
- `App.tsx` (clean import of removed components — already done in Task 5)

**Step 1: Remove Sidebar and containers**

```bash
rm apps/desktop/src/components/Sidebar.tsx
rm apps/desktop/src/components/containers/ImportContainer.tsx
rm apps/desktop/src/components/containers/AnalyzeContainer.tsx
rm apps/desktop/src/components/containers/ReviewContainer.tsx
rm apps/desktop/src/components/containers/GenerateContainer.tsx
rm apps/desktop/src/components/containers/DoneContainer.tsx
```

**Step 2: Check for remaining references**

Run: `cd apps/desktop && npx tsc --noEmit`
Expected: No errors (or only pre-existing ones).

**Step 3: Run tests**

Run: `cd apps/desktop && npm test -- --run`
Expected: All tests pass. Fix any that reference removed components.

**Step 4: Commit**

```bash
git add -A apps/desktop/src/
git commit -m "refactor: remove old sidebar and container components replaced by library/book-detail"
```

---

### Task 8: Add CSS Styles for Library & Book Detail

**Files:**
- Modify: `apps/desktop/src/styles.css`

**Step 1: Add new CSS**

Append to styles.css:

```css
/* ── Library View ────────────────────────────────────── */
.library-view { padding: 2rem; }
.library-empty { text-align: center; margin-top: 4rem; }
.library-empty h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.library-empty p { color: #888; margin-bottom: 1.5rem; }
.library-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.library-header h1 { font-size: 1.75rem; font-weight: 700; }
.library-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
.library-card {
  background: var(--card-bg, #1a1a2e); border: 1px solid var(--border, #333);
  border-radius: 8px; padding: 1rem; cursor: pointer; text-align: center;
  transition: border-color 0.15s; width: 100%;
}
.library-card:hover { border-color: var(--primary, #6c63ff); }
.card-cover { font-size: 2.5rem; margin-bottom: 0.5rem; }
.card-title { font-weight: 600; margin-bottom: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-date { font-size: 0.75rem; color: #666; }

.progress-bar { width: 100%; height: 6px; background: #333; border-radius: 3px; overflow: hidden; margin-bottom: 0.25rem; }
.progress-fill { height: 100%; background: var(--primary, #6c63ff); border-radius: 3px; transition: width 0.3s; }
.progress-text { font-size: 0.75rem; color: #888; }

/* ── Book Detail ─────────────────────────────────────── */
.book-detail { display: flex; flex-direction: column; height: 100vh; }
.detail-header { display: flex; align-items: center; gap: 1rem; padding: 1rem 2rem; border-bottom: 1px solid var(--border, #333); }
.detail-header h1 { font-size: 1.25rem; font-weight: 600; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.btn-back { background: none; border: none; color: var(--primary, #6c63ff); cursor: pointer; font-size: 0.9rem; }

.detail-body { display: flex; flex: 1; overflow: hidden; }

/* Chapter list (left panel) */
.chapter-list { width: 260px; padding: 1rem; border-right: 1px solid var(--border, #333); overflow-y: auto; flex-shrink: 0; }
.chapter-list h3 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin-bottom: 0.75rem; }
.select-all { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; font-size: 0.85rem; cursor: pointer; }
.chapter-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; font-size: 0.85rem; cursor: pointer; }
.chapter-item input[type="checkbox"] { flex-shrink: 0; }
.chapter-status { width: 1.5rem; text-align: center; flex-shrink: 0; }
.chapter-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Detail content (right panel) */
.detail-content { flex: 1; padding: 1.5rem 2rem; overflow-y: auto; }
.detail-tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border, #333); }
.tab-btn { background: none; border: none; padding: 0.5rem 1.25rem; cursor: pointer; color: #888; font-size: 0.9rem; border-bottom: 2px solid transparent; transition: all 0.15s; }
.tab-btn.active { color: var(--primary, #6c63ff); border-bottom-color: var(--primary, #6c63ff); }
.tab-btn:hover:not(.active) { color: #ccc; }
.tab-panel { min-height: 200px; }

.character-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.character-table th { text-align: left; padding: 0.5rem; border-bottom: 1px solid var(--border, #333); color: #888; font-weight: 500; }
.character-table td { padding: 0.5rem; border-bottom: 1px solid var(--border, #333); }
.character-table select { background: var(--bg, #111); color: var(--fg, #fff); border: 1px solid var(--border, #333); padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; }

.audio-row { display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0; }
.audio-row span { min-width: 200px; }
.audio-row audio { height: 32px; }

.character-strip {
  padding: 0.5rem 2rem; border-top: 1px solid var(--border, #333); font-size: 0.8rem; color: #888; cursor: pointer;
  display: flex; align-items: center; gap: 0.5rem; overflow-x: auto;
}
.character-strip:hover { color: var(--primary, #6c63ff); }
.character-chip {
  display: inline-block; padding: 0.15rem 0.5rem; background: var(--card-bg, #1a1a2e); border: 1px solid var(--border, #333);
  border-radius: 12px; font-size: 0.75rem;
}

.error-banner { background: #3a1010; color: #ff6b6b; padding: 0.75rem 2rem; display: flex; justify-content: space-between; align-items: center; }
.error-text { color: #ff6b6b; margin-top: 0.5rem; }
.success-text { color: #4caf50; margin-top: 0.5rem; }

.chapter-status-row { font-size: 0.8rem; padding: 0.2rem 0; color: #888; }

.btn-primary {
  background: var(--primary, #6c63ff); color: #fff; border: none; padding: 0.5rem 1.25rem;
  border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 500;
}
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-secondary {
  background: var(--card-bg, #1a1a2e); color: var(--fg, #fff); border: 1px solid var(--border, #333);
  padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
}
```

**Step 2: Verify no broken styles**

Run: `cd apps/desktop && npx tsc --noEmit` (CSS doesn't affect typechecking but check overall)

**Step 3: Commit**

```bash
git add apps/desktop/src/styles.css
git commit -m "style: add library and book detail CSS"
```

---

### Task 9: Use Persistent App Data Directory for WorkDirs

**Files:**
- Modify: `apps/desktop/src/App.tsx` (import flow — change workDir from tempDir to app data)
- Modify: `apps/desktop/src-tauri/src/lib.rs` (add command to get persistent dir)

**Step 1: Add Rust command for persistent books directory**

In `lib.rs`, add:

```rust
#[tauri::command]
fn book_work_dir(book_id: &str) -> String {
    dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("audiobook-generator")
        .join("books")
        .join(book_id)
        .to_str()
        .unwrap()
        .to_string()
}
```

Register in `invoke_handler`.

**Step 2: Add store helper for work dir**

In `store.ts`, add:

```ts
bookWorkDir(bookId: string): Promise<string> {
  return invoke("book_work_dir", { bookId }) as Promise<string>;
},
```

**Step 3: Update App.tsx import flow to use persistent dir**

Change the import flow in App.tsx's `handleImport` to use `db.bookWorkDir(bookId)` instead of `tempDir()`. Also update the `tempDir` import usage — keep it only as fallback if needed, but prefer the persistent path:

```tsx
// In handleImport:
const workDir = await db.bookWorkDir(bookId);
```

**Step 4: Remove tempDir import from App.tsx** (if no longer needed)

**Step 5: Build and verify**

Run: `cd apps/desktop/src-tauri && cargo check`

**Step 6: Commit**

```bash
git add apps/desktop/src-tauri/src/lib.rs apps/desktop/src/state/store.ts apps/desktop/src/App.tsx
git commit -m "feat: persist book work dirs in app data directory"
```

---

### Task 10: Load Saved State on Book Detail Open

**Files:**
- Modify: `apps/desktop/src/components/BookDetailView.tsx`

**Step 1: Add `useEffect` to restore saved state when book loads**

In `BookDetailView`, add a `useEffect` that:
1. Loads characters from DB and seeds `analysis` state
2. Loads chapters with scripts from DB and populates `analysis.scriptPaths`
3. Loads existing audio paths

```tsx
useEffect(() => {
  let cancelled = false;
  async function restore() {
    // Restore characters
    const chars = await db.getCharacters(book.bookId);
    if (cancelled) return;
    if (chars.length > 0) {
      setAnalysis(prev => ({
        characters: chars.map((c): CharacterMeta => ({
          id: c.id,
          canonicalName: c.canonicalName,
          aliases: JSON.parse(c.aliases || "[]"),
          gender: c.gender || "unknown",
          voiceId: c.voiceId || "narrator_default",
          confidence: c.confidence,
        })),
        voices: prev?.voices ?? [],
        scriptPaths: prev?.scriptPaths ?? {},
      }));
    }

    // Restore chapters with scripts
    const chaptersWithScripts = await db.getChaptersWithScripts(book.bookId);
    if (cancelled) return;
    const scriptPaths: Record<string, string> = {};
    for (const ch of chaptersWithScripts) {
      scriptPaths[ch.id] = ch.scriptPath;
    }
    if (Object.keys(scriptPaths).length > 0) {
      setAnalysis(prev => prev ? { ...prev, scriptPaths: { ...prev.scriptPaths, ...scriptPaths } } : { characters: [], voices: [], scriptPaths });
    }

    // Check for existing audio files
    const audioPaths: Record<string, string> = {};
    for (const ch of book.chapters) {
      const audioPath = `${book.workDir}/audio/${ch.id}.wav`;
      try {
        await invoke("run_worker", { command: "_read_file", inputJson: JSON.stringify({ path: audioPath }) });
        audioPaths[ch.id] = audioPath;
      } catch { /* file doesn't exist */ }
    }
    if (!cancelled) setChapterAudioPaths(audioPaths);
  }
  restore();
  return () => { cancelled = true; };
}, [book.bookId, book.workDir, book.chapters]);
```

**Step 2: Typecheck**

Run: `cd apps/desktop && npx tsc --noEmit`

**Step 3: Commit**

```bash
git add apps/desktop/src/components/BookDetailView.tsx
git commit -m "feat: restore saved state (characters, scripts, audio) on book detail open"
```

---

### Task 11: Final Integration & Verification

**Step 1: Run full typecheck**

Run: `cd apps/desktop && npx tsc --noEmit`

**Step 2: Run tests**

Run: `cd apps/desktop && npm test -- --run`

**Step 3: Run Rust build check**

Run: `cd apps/desktop/src-tauri && cargo check`

**Step 4: Fix any issues found**

**Step 5: Run linter**

Run: `cd apps/desktop && npx tsc --noEmit`
Run: `cd workers/python && uv run ruff check .`

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete library feature — persistent dup-detect, cumulative characters, book detail"
```
