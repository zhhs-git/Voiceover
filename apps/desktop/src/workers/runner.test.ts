// @vitest-environment node
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { runWorkerCommand } from "./runner";

describe("runWorkerCommand", () => {
  test("writes JSON input and parses structured worker output", async () => {
    const directory = await mkdtemp(join(tmpdir(), "worker-runner-"));
    const scriptPath = join(directory, "worker.mjs");
    await writeFile(
      scriptPath,
      `
        import { readFile, writeFile } from "node:fs/promises";
        const [, , command, inputPath, outputPath] = process.argv;
        const input = JSON.parse(await readFile(inputPath, "utf-8"));
        await writeFile(outputPath, JSON.stringify({
          status: "succeeded",
          warnings: [],
          artifacts: [{ kind: command, path: input.inputPath }]
        }));
      `,
      "utf-8",
    );

    const result = await runWorkerCommand({
      executable: process.execPath,
      argsPrefix: [scriptPath],
      command: "extract_book_text",
      input: { inputPath: "/tmp/book.epub" },
      workDirectory: directory,
      timeoutMs: 2_000,
    });

    expect(result.status).toBe("succeeded");
    expect(result.artifacts[0].kind).toBe("extract_book_text");
  });

  test("returns a structured failure when the worker exits without output", async () => {
    const directory = await mkdtemp(join(tmpdir(), "worker-runner-"));
    const scriptPath = join(directory, "worker.mjs");
    await writeFile(scriptPath, "process.exit(7);", "utf-8");

    const result = await runWorkerCommand({
      executable: process.execPath,
      argsPrefix: [scriptPath],
      command: "extract_book_text",
      input: {},
      workDirectory: directory,
      timeoutMs: 2_000,
    });

    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("worker_exit_failed");
  });
});
