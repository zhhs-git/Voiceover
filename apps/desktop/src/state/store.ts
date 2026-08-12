import { invoke } from "../lib/platform";
import type { LibraryBook, CharacterRow, ChapterRow } from "../types";

export function createAudiobookStore() {
  return {
    createBook(record: { id: string; title: string; sourcePath: string; workDir: string }) {
      return invoke("db_create_book", { id: record.id, title: record.title, sourcePath: record.sourcePath, workDir: record.workDir });
    },
    deleteBook(bookId: string) {
      return invoke("db_delete_book", { bookId });
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
    async setNarratorVoice(
      bookId: string,
      narratorVoiceId: "narrator_female" | "narrator_male" | "narrator_default",
    ): Promise<void> {
      await invoke("db_set_narrator_voice", { bookId, narratorVoiceId });
    },
    async upsertCharacter(record: {
      id: string; bookId: string; canonicalName: string;
      gender?: string | null; ageClass?: string | null;
      identityStatus?: "provisional" | "confirmed" | "merged" | null;
      voiceId?: string | null; voiceSource?: "auto" | "manual" | null;
      voiceAssignmentVersion?: number | null; voiceProfile?: string | null;
      fallbackVoiceId?: string | null;
      voiceDesign?: string | null;
      voiceDescription?: string | null;
      confidence?: number; aliases?: string;
    }) {
      return invoke("db_upsert_character", {
        id: record.id, bookId: record.bookId,
        canonicalName: record.canonicalName,
        gender: record.gender ?? null,
        ageClass: record.ageClass ?? null,
        identityStatus: record.identityStatus ?? null,
        voiceId: record.voiceId ?? null,
        voiceSource: record.voiceSource ?? null,
        voiceAssignmentVersion: record.voiceAssignmentVersion ?? null,
        voiceProfile: record.voiceProfile ?? null,
        fallbackVoiceId: record.fallbackVoiceId ?? null,
        voiceDesign: record.voiceDesign ?? null,
        voiceDescription: record.voiceDescription ?? null,
        confidence: record.confidence ?? 0.0,
        aliases: record.aliases ?? "[]",
      });
    },
    async getCharacters(bookId: string): Promise<CharacterRow[]> {
      return await invoke("db_get_characters", { bookId }) as any;
    },
    async getChapters(bookId: string): Promise<ChapterRow[]> {
      return await invoke("db_get_chapters", { bookId }) as any;
    },
    async bookWorkDir(bookId: string): Promise<string> {
      return await invoke("book_work_dir", { bookId }) as string;
    },
  };
}
