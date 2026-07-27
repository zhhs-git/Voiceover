export const SUPPORTED_BOOK_EXTENSIONS = ["epub", "pdf", "txt"] as const;

export function isSupportedBookPath(path: string): boolean {
  const extension = path.split(".").pop()?.toLowerCase();
  return SUPPORTED_BOOK_EXTENSIONS.some((supported) => supported === extension);
}
