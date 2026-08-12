use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use std::process::Command;
use std::sync::{mpsc, Mutex};

// ── Shared DB state ────────────────────────────────────────────────────────

struct Db(Mutex<Connection>);

fn get_db() -> &'static Db {
    static DB: std::sync::OnceLock<Db> = std::sync::OnceLock::new();
    DB.get_or_init(|| {
        let db_path = dirs::config_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("audiobook-generator")
            .join("audiobook.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).ok();
        let conn = Connection::open(&db_path).expect("failed to open database");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
                source_language TEXT NOT NULL, output_language TEXT NOT NULL, work_dir TEXT NOT NULL,
                imported_at TEXT, updated_at TEXT,
                narrator_voice_id TEXT NOT NULL DEFAULT 'narrator_female'
            );
            CREATE TABLE IF NOT EXISTS chapters (
                id TEXT NOT NULL, book_id TEXT NOT NULL, title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', script_path TEXT,
                PRIMARY KEY (id, book_id)
            );
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT NOT NULL, book_id TEXT NOT NULL, canonical_name TEXT NOT NULL,
                gender TEXT, age_class TEXT, identity_status TEXT DEFAULT 'confirmed', voice_id TEXT, voice_source TEXT,
                voice_assignment_version INTEGER, voice_profile TEXT, fallback_voice_id TEXT, voice_design TEXT, voice_description TEXT,
                confidence REAL DEFAULT 0.0,
                aliases TEXT DEFAULT '[]', updated_at TEXT,
                PRIMARY KEY (id, book_id)
            );
            CREATE TABLE IF NOT EXISTS character_aliases (
                book_id TEXT NOT NULL, character_id TEXT NOT NULL,
                alias_key TEXT NOT NULL, alias TEXT NOT NULL, updated_at TEXT,
                PRIMARY KEY (book_id, character_id, alias_key)
            );
            CREATE INDEX IF NOT EXISTS idx_books_source_path ON books(source_path);
            CREATE INDEX IF NOT EXISTS idx_characters_book_id ON characters(book_id);
            CREATE INDEX IF NOT EXISTS idx_character_aliases_lookup ON character_aliases(book_id, alias_key);
            CREATE INDEX IF NOT EXISTS idx_chapters_book_id ON chapters(book_id);",
        )
        .expect("failed to create tables");
        // Add columns to existing databases that lack them
        let has_imported_at = conn
            .prepare("SELECT imported_at FROM books LIMIT 1")
            .is_ok();
        if !has_imported_at {
            conn.execute("ALTER TABLE books ADD COLUMN imported_at TEXT", [])
                .expect("failed to add imported_at column");
            conn.execute("ALTER TABLE books ADD COLUMN updated_at TEXT", [])
                .expect("failed to add updated_at column");
        }
        for (column, definition) in [
            ("age_class", "TEXT"),
            ("identity_status", "TEXT"),
            ("voice_source", "TEXT"),
            ("voice_assignment_version", "INTEGER"),
            ("voice_profile", "TEXT"),
            ("fallback_voice_id", "TEXT"),
            ("voice_design", "TEXT"),
            ("voice_description", "TEXT"),
        ] {
            let has_column = conn
                .prepare(&format!("SELECT {column} FROM characters LIMIT 1"))
                .is_ok();
            if !has_column {
                conn.execute(
                    &format!("ALTER TABLE characters ADD COLUMN {column} {definition}"),
                    [],
                )
                .unwrap_or_else(|_| panic!("failed to add {column} column"));
            }
        }
        let has_narrator_voice_id = conn
            .prepare("SELECT narrator_voice_id FROM books LIMIT 1")
            .is_ok();
        if !has_narrator_voice_id {
            conn.execute(
                "ALTER TABLE books ADD COLUMN narrator_voice_id TEXT NOT NULL DEFAULT 'narrator_female'",
                [],
            )
            .expect("failed to add narrator_voice_id column");
        }
        Db(Mutex::new(conn))
    })
}

// ── Book commands ───────────────────────────────────────────────────────────

#[tauri::command]
fn db_create_book(
    id: String,
    title: String,
    source_path: String,
    work_dir: String,
) -> Result<(), String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let now = Utc::now().to_rfc3339();
    db.execute(
        "INSERT OR REPLACE INTO books (id, title, source_path, source_language, output_language, work_dir, imported_at, updated_at, narrator_voice_id) VALUES (?1, ?2, ?3, 'en', 'en', ?4, ?5, ?5, 'narrator_female')",
        rusqlite::params![id, title, source_path, work_dir, now],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn db_delete_book(book_id: String) -> Result<(), String> {
    let (work_dir, deleted) = {
        let db = get_db().0.lock().map_err(|e| e.to_string())?;
        let work_dir = db
            .query_row(
                "SELECT work_dir FROM books WHERE id = ?1",
                rusqlite::params![book_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let Some(work_dir) = work_dir else {
            return Ok(());
        };

        let tx = db.unchecked_transaction().map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM characters WHERE book_id = ?1",
            rusqlite::params![book_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM character_aliases WHERE book_id = ?1",
            rusqlite::params![book_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM chapters WHERE book_id = ?1",
            rusqlite::params![book_id],
        )
        .map_err(|e| e.to_string())?;
        let deleted = tx
            .execute(
                "DELETE FROM books WHERE id = ?1",
                rusqlite::params![book_id],
            )
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        (work_dir, deleted)
    };

    if deleted > 0 {
        let path = std::path::Path::new(&work_dir);
        if path.exists() {
            std::fs::remove_dir_all(path).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[tauri::command]
fn db_set_narrator_voice(book_id: String, narrator_voice_id: String) -> Result<(), String> {
    let normalized = match narrator_voice_id.as_str() {
        "narrator_male" => "narrator_male",
        "narrator_female" => "narrator_female",
        "narrator_default" => "narrator_default",
        _ => return Err("invalid narrator voice id".to_string()),
    };
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let work_dir = db
        .query_row(
            "SELECT work_dir FROM books WHERE id = ?1",
            rusqlite::params![book_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let Some(work_dir) = work_dir else {
        return Err("book not found".to_string());
    };
    db.execute(
        "UPDATE books SET narrator_voice_id = ?1, updated_at = ?2 WHERE id = ?3",
        rusqlite::params![normalized, Utc::now().to_rfc3339(), book_id],
    )
    .map_err(|e| e.to_string())?;
    drop(db);

    let audio_dir = std::path::Path::new(&work_dir).join("audio");
    if let Ok(entries) = std::fs::read_dir(audio_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("wav") {
                std::fs::remove_file(path).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn db_upsert_chapter(
    id: String,
    book_id: String,
    title: String,
    status: String,
    script_path: Option<String>,
) -> Result<(), String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    db.execute(
        "INSERT OR REPLACE INTO chapters (id, book_id, title, status, script_path) VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![id, book_id, title, status, script_path],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn db_get_chapters_with_scripts(book_id: String) -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, script_path FROM chapters WHERE book_id = ?1 AND script_path IS NOT NULL ORDER BY id ASC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![book_id], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "scriptPath": row.get::<_, String>(1)?
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
fn db_list_books() -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, title, source_path, work_dir, imported_at, narrator_voice_id FROM books ORDER BY imported_at DESC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "title": row.get::<_, String>(1)?,
                "sourcePath": row.get::<_, String>(2)?,
                "workDir": row.get::<_, String>(3)?,
                "importedAt": row.get::<_, Option<String>>(4)?,
                "narratorVoiceId": row.get::<_, Option<String>>(5)?
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
        .prepare("SELECT id, title, source_path, work_dir, imported_at, narrator_voice_id FROM books WHERE source_path = ?1")
        .map_err(|e| e.to_string())?;
    let mut rows = stmt
        .query_map(rusqlite::params![source_path], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "title": row.get::<_, String>(1)?,
                "sourcePath": row.get::<_, String>(2)?,
                "workDir": row.get::<_, String>(3)?,
                "importedAt": row.get::<_, Option<String>>(4)?,
                "narratorVoiceId": row.get::<_, Option<String>>(5)?
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
    id: String,
    book_id: String,
    canonical_name: String,
    gender: Option<String>,
    age_class: Option<String>,
    identity_status: Option<String>,
    voice_id: Option<String>,
    voice_source: Option<String>,
    voice_assignment_version: Option<i64>,
    voice_profile: Option<String>,
    fallback_voice_id: Option<String>,
    voice_design: Option<String>,
    voice_description: Option<String>,
    confidence: Option<f64>,
    aliases: Option<String>,
) -> Result<(), String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let now = Utc::now().to_rfc3339();
    let incoming_aliases = aliases.clone().unwrap_or_else(|| "[]".to_string());
    let incoming_names = character_names(&canonical_name, &incoming_aliases);
    let exact_id = db
        .query_row(
            "SELECT id FROM characters WHERE book_id = ?1 AND id = ?2",
            rusqlite::params![book_id, id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    let matching_id = if exact_id.is_some() {
        exact_id
    } else {
        let mut stmt = db
            .prepare("SELECT id, canonical_name, aliases FROM characters WHERE book_id = ?1")
            .map_err(|e| e.to_string())?;
        let mut found: Option<String> = None;
        let rows = stmt
            .query_map(rusqlite::params![book_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (existing_id, existing_name, existing_aliases) = row.map_err(|e| e.to_string())?;
            let existing_names =
                character_names(&existing_name, existing_aliases.as_deref().unwrap_or("[]"));
            if incoming_names
                .iter()
                .any(|name| existing_names.contains(name))
            {
                found = Some(existing_id);
                break;
            }
        }
        found
    };

    let target_id = matching_id.unwrap_or(id);
    db.execute(
        "INSERT INTO characters (id, book_id, canonical_name, gender, age_class, identity_status, voice_id, voice_source, voice_assignment_version, voice_profile, fallback_voice_id, voice_design, voice_description, confidence, aliases, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
         ON CONFLICT(id, book_id) DO UPDATE SET
           canonical_name = excluded.canonical_name,
           gender = COALESCE(excluded.gender, characters.gender),
           age_class = COALESCE(excluded.age_class, characters.age_class),
           identity_status = CASE
             WHEN excluded.identity_status = 'confirmed' THEN 'confirmed'
             ELSE COALESCE(excluded.identity_status, characters.identity_status)
           END,
           voice_id = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_id
             ELSE COALESCE(excluded.voice_id, characters.voice_id)
           END,
           voice_source = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_source
             ELSE COALESCE(excluded.voice_source, characters.voice_source)
           END,
           voice_assignment_version = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_assignment_version
             WHEN excluded.voice_source = 'manual' THEN NULL
             ELSE COALESCE(excluded.voice_assignment_version, characters.voice_assignment_version)
           END,
           voice_profile = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_profile
             WHEN excluded.voice_source = 'manual' THEN NULL
             ELSE COALESCE(excluded.voice_profile, characters.voice_profile)
           END,
           fallback_voice_id = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.fallback_voice_id
             WHEN excluded.voice_source = 'manual' THEN NULL
             ELSE COALESCE(excluded.fallback_voice_id, characters.fallback_voice_id)
           END,
           voice_design = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_design
             ELSE COALESCE(excluded.voice_design, characters.voice_design)
           END,
           voice_description = CASE
             WHEN characters.voice_source = 'manual' AND excluded.voice_source <> 'manual' THEN characters.voice_description
             WHEN excluded.voice_source = 'manual' THEN excluded.voice_description
             ELSE COALESCE(excluded.voice_description, characters.voice_description)
           END,
           confidence = COALESCE(excluded.confidence, characters.confidence),
           aliases = excluded.aliases,
           updated_at = excluded.updated_at",
        rusqlite::params![target_id, book_id, canonical_name, gender, age_class, identity_status, voice_id, voice_source, voice_assignment_version, voice_profile, fallback_voice_id, voice_design, voice_description, confidence, incoming_aliases, now],
    ).map_err(|e| e.to_string())?;
    db.execute(
        "DELETE FROM character_aliases WHERE book_id = ?1 AND character_id = ?2",
        rusqlite::params![book_id, target_id],
    )
    .map_err(|e| e.to_string())?;
    for alias in character_names(&canonical_name, &incoming_aliases) {
        db.execute(
            "INSERT OR IGNORE INTO character_aliases (book_id, character_id, alias_key, alias, updated_at) VALUES (?1, ?2, ?3, ?3, ?4)",
            rusqlite::params![book_id, target_id, alias, now],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn character_names(canonical_name: &str, aliases_json: &str) -> Vec<String> {
    let mut names = vec![canonical_name.trim().to_lowercase()];
    if let Ok(aliases) = serde_json::from_str::<Vec<String>>(aliases_json) {
        names.extend(aliases.into_iter().map(|alias| alias.trim().to_lowercase()));
    }
    names.retain(|name| !name.is_empty());
    names.retain(|name| !is_generic_character_label(name));
    names.sort();
    names.dedup();
    names
}

fn is_generic_character_label(value: &str) -> bool {
    matches!(
        value,
        "小姐"
            | "少爷"
            | "姑娘"
            | "公子"
            | "夫人"
            | "太太"
            | "老爷"
            | "殿下"
            | "陛下"
            | "皇上"
            | "皇后"
            | "公主"
            | "王爷"
            | "世子"
            | "大人"
            | "先生"
            | "女士"
            | "母亲"
            | "父亲"
            | "娘"
            | "爹"
            | "妈妈"
            | "爸爸"
            | "mother"
            | "father"
            | "wife"
            | "husband"
            | "miss"
            | "mrs"
            | "ms"
            | "mr"
            | "sir"
            | "madam"
            | "lady"
            | "lord"
            | "girl"
            | "boy"
            | "woman"
            | "man"
    )
}

#[tauri::command]
fn db_get_characters(book_id: String) -> Result<Vec<serde_json::Value>, String> {
    let db = get_db().0.lock().map_err(|e| e.to_string())?;
    let mut stmt = db
        .prepare("SELECT id, canonical_name, gender, age_class, identity_status, voice_id, voice_source, voice_assignment_version, voice_profile, fallback_voice_id, voice_design, voice_description, confidence, aliases FROM characters WHERE book_id = ?1")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map(rusqlite::params![book_id], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "canonicalName": row.get::<_, String>(1)?,
                "gender": row.get::<_, Option<String>>(2)?,
                "ageClass": row.get::<_, Option<String>>(3)?,
                "identityStatus": row.get::<_, Option<String>>(4)?,
                "voiceId": row.get::<_, Option<String>>(5)?,
                "voiceSource": row.get::<_, Option<String>>(6)?,
                "voiceAssignmentVersion": row.get::<_, Option<i64>>(7)?,
                "voiceProfile": row.get::<_, Option<String>>(8)?,
                "fallbackVoiceId": row.get::<_, Option<String>>(9)?,
                "voiceDesign": row.get::<_, Option<String>>(10)?,
                "voiceDescription": row.get::<_, Option<String>>(11)?,
                "confidence": row.get::<_, Option<f64>>(12)?,
                "aliases": row.get::<_, Option<String>>(13)?
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
        .prepare("SELECT id, title, status, script_path FROM chapters WHERE book_id = ?1 ORDER BY id ASC")
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

#[tauri::command]
fn book_work_dir(book_id: String) -> String {
    dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("audiobook-generator")
        .join("books")
        .join(&book_id)
        .to_str()
        .unwrap()
        .to_string()
}

// ── Worker command ──────────────────────────────────────────────────────────

#[tauri::command]
async fn run_worker(command: String, input_json: String) -> Result<String, String> {
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join(format!("audiobook-{}-input.json", command));
    let output_path = temp_dir.join(format!("audiobook-{}-output.json", command));

    std::fs::write(&input_path, &input_json).map_err(|e| e.to_string())?;

    let worker_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("workers")
        .join("python");

    let python = worker_dir.join(".venv").join("bin").join("python3");
    let input = input_path.to_str().unwrap().to_string();
    let output = output_path.to_str().unwrap().to_string();

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = Command::new(&python)
            .args(["-m", "audiobook_worker.cli", &command, &input, &output])
            .current_dir(&worker_dir)
            .env_remove("PYTHONPATH")
            .env("AUDIOBOOK_TTS_DEVICE", "mps")
            .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            .output();
        let _ = tx.send(result);
    });

    let result = rx
        .recv()
        .map_err(|_| "Worker thread panicked".to_string())?
        .map_err(|e| format!("Failed to spawn worker: {}", e))?;

    if let Ok(output) = std::fs::read_to_string(&output_path) {
        return Ok(output);
    }
    Err(format!(
        "Worker exited {:?}: {}",
        result.status.code(),
        String::from_utf8_lossy(&result.stderr)
    ))
}

#[tauri::command]
async fn copy_file(from: String, to: String) -> Result<String, String> {
    std::fs::copy(&from, &to).map_err(|e| e.to_string())?;
    Ok(to)
}

#[tauri::command]
fn file_exists(paths: Vec<String>) -> Vec<String> {
    paths
        .into_iter()
        .filter(|p| std::path::Path::new(p).exists())
        .collect()
}

// ── App entry ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = get_db(); // init DB on startup
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            run_worker,
            copy_file,
            file_exists,
            db_create_book,
            db_delete_book,
            db_upsert_chapter,
            db_get_chapters_with_scripts,
            book_work_dir,
            db_list_books,
            db_get_book,
            db_set_narrator_voice,
            db_upsert_character,
            db_get_characters,
            db_get_chapters,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
