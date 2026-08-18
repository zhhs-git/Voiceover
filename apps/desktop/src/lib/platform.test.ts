import { afterEach, describe, expect, test, vi } from "vitest";

import { downloadFinalAudioArchive } from "./platform";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("downloadFinalAudioArchive", () => {
  test("posts an MP3 export request and uses the streamed attachment filename", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({
        "Content-Disposition": "attachment; filename*=UTF-8''%E7%A4%BA%E4%BE%8B.zip",
        "X-Audiobook-Chapter-Count": "2",
        "X-Audiobook-Skipped-Chapter-Count": "1",
      }),
      blob: vi.fn().mockResolvedValue(new Blob(["zip"])),
    });
    const createObjectURL = vi.fn().mockReturnValue("blob:final-audio");
    const revokeObjectURL = vi.fn();
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    const result = await downloadFinalAudioArchive({
      bookId: "book 123",
      chapterIds: ["chapter_001", "chapter_002"],
      format: "mp3",
      bitrateKbps: 256,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/books/book%20123/final-audio.zip",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          chapterIds: ["chapter_001", "chapter_002"],
          format: "mp3",
          bitrateKbps: 256,
        }),
      }),
    );
    expect(anchorClick).toHaveBeenCalledOnce();
    expect(result).toEqual({ filename: "示例.zip", chapterCount: 2, skippedCount: 1 });
  });
});
