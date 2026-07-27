import { describe, expect, it } from "vitest";
import { isSupportedBookPath } from "./bookFormats";

describe("isSupportedBookPath", () => {
  it("accepts EPUB, PDF, and TXT paths case-insensitively", () => {
    expect(isSupportedBookPath("/books/story.epub")).toBe(true);
    expect(isSupportedBookPath("/books/story.PDF")).toBe(true);
    expect(isSupportedBookPath("/books/中文小说.TXT")).toBe(true);
  });

  it("rejects unsupported files", () => {
    expect(isSupportedBookPath("/books/audio.mp3")).toBe(false);
    expect(isSupportedBookPath("/books/no-extension")).toBe(false);
  });
});
