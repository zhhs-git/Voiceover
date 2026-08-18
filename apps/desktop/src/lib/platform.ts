export interface WebInvokeArgs {
  [key: string]: unknown;
}

export type FinalAudioExportFormat = "mp3" | "wav";

export interface FinalAudioArchiveDownload {
  filename: string;
  chapterCount: number;
  skippedCount: number;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = (await response.json().catch(() => ({}))) as {
    error?: string;
    value?: T;
  };
  if (!response.ok) {
    throw new Error(payload.error ?? `请求失败（${response.status}）`);
  }
  return ("value" in payload ? payload.value : payload) as T;
}

/** Browser replacement for the old Tauri invoke bridge. */
export async function invoke<T = unknown>(command: string, args: WebInvokeArgs = {}): Promise<T> {
  return request<T>("/api/invoke", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command, args }),
  });
}

export function convertFileSrc(path: string): string {
  return `/api/files?path=${encodeURIComponent(path)}`;
}

export function selectBookFile(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".epub,.pdf,.txt,application/epub+zip,application/pdf,text/plain";
    document.body.appendChild(input);
    const finish = (file: File | null) => {
      input.remove();
      resolve(file);
    };
    input.onchange = () => finish(input.files?.[0] ?? null);
    input.oncancel = () => finish(null);
    input.click();
  });
}

export async function importBook(file: File): Promise<Record<string, unknown>> {
  if (file.size === 0) {
    throw new Error("浏览器读取到的文件大小为 0，请重新选择已下载到本机的文件");
  }
  const form = new FormData();
  form.append("file", file, file.name);
  const response = await fetch(`/api/books/import?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    body: form,
  });
  const payload = (await response.json()) as Record<string, unknown>;
  if (!response.ok) {
    const error = payload.error as string | { message?: string } | undefined;
    const message = typeof error === "string" ? error : error?.message;
    throw new Error(message ?? `导入失败（${response.status}）`);
  }
  return payload;
}

export async function readChapterText(
  bookId: string,
  chapterId: string,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(
    `/api/books/${encodeURIComponent(bookId)}/chapters/${encodeURIComponent(chapterId)}/text`,
    { signal },
  );
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as {
      error?: string | { message?: string };
    };
    const error = payload.error;
    const message = typeof error === "string" ? error : error?.message;
    throw new Error(message ?? `读取章节失败（${response.status}）`);
  }
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? "";
  if (!contentType.startsWith("text/plain")) {
    throw new Error("章节接口返回的不是文本内容，请重启主机服务后重试");
  }
  return response.text();
}

export function downloadFile(url: string, filename: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function downloadFilename(response: Response, fallback: string): string {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  if (encoded) {
    try {
      return decodeURIComponent(encoded);
    } catch {
      // Fall through to the ASCII filename or the caller's safe fallback.
    }
  }
  return disposition.match(/filename="([^"]+)"/i)?.[1] ?? fallback;
}

function responseCount(response: Response, header: string): number {
  const parsed = Number(response.headers.get(header));
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : 0;
}

export async function downloadFinalAudioArchive({
  bookId,
  chapterIds,
  format,
  bitrateKbps,
}: {
  bookId: string;
  chapterIds: string[];
  format: FinalAudioExportFormat;
  bitrateKbps?: number;
}): Promise<FinalAudioArchiveDownload> {
  const response = await fetch(
    `/api/books/${encodeURIComponent(bookId)}/final-audio.zip`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chapterIds,
        format,
        ...(format === "mp3" ? { bitrateKbps } : {}),
      }),
    },
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as {
      error?: string | { message?: string };
    };
    const error = payload.error;
    const message = typeof error === "string" ? error : error?.message;
    throw new Error(message ?? `打包下载失败（${response.status}）`);
  }

  const filename = downloadFilename(
    response,
    `final-audio-${format}${format === "mp3" ? `-${bitrateKbps ?? 192}kbps` : ""}.zip`,
  );
  const blobUrl = URL.createObjectURL(await response.blob());
  downloadFile(blobUrl, filename);
  window.setTimeout(() => {
    if (typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(blobUrl);
  }, 0);
  return {
    filename,
    chapterCount: responseCount(response, "X-Audiobook-Chapter-Count"),
    skippedCount: responseCount(response, "X-Audiobook-Skipped-Chapter-Count"),
  };
}
