import type { BookState } from "../types";

interface ExtractionCache {
  sourcePath: string;
  book: BookState;
}

type ReadJson = (path: string) => Promise<unknown>;
type WriteJson = (path: string, payload: ExtractionCache) => Promise<void>;

export function extractionCachePath(workDir: string): string {
  return `${workDir}/book-extraction.json`;
}

export async function cachedBookFromExtraction({
  cachePath,
  sourcePath,
  readJson,
}: {
  cachePath: string;
  sourcePath: string;
  readJson: ReadJson;
}): Promise<BookState | null> {
  try {
    const payload = (await readJson(cachePath)) as Partial<ExtractionCache>;
    if (payload.sourcePath !== sourcePath || !payload.book) return null;
    return payload.book;
  } catch {
    return null;
  }
}

export async function writeExtractionCache({
  sourcePath,
  book,
  writeJson,
}: {
  sourcePath: string;
  book: BookState;
  writeJson: WriteJson;
}): Promise<void> {
  await writeJson(extractionCachePath(book.workDir), {
    sourcePath,
    book,
  });
}
