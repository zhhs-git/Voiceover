import { spawn } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export interface WorkerRunOptions {
  executable: string;
  argsPrefix?: string[];
  command: string;
  input: Record<string, unknown>;
  workDirectory?: string;
  timeoutMs: number;
}

export interface WorkerResponse {
  status: "pending" | "running" | "succeeded" | "failed" | "skipped" | "needs_review";
  warnings: string[];
  artifacts: Array<{ kind: string; path: string; metadata?: Record<string, unknown> }>;
  error?: { code: string; message: string; details?: Record<string, unknown> };
}

export async function runWorkerCommand(options: WorkerRunOptions): Promise<WorkerResponse> {
  const runDirectory = options.workDirectory ?? (await mkdtemp(join(tmpdir(), "audiobook-worker-")));
  const inputPath = join(runDirectory, `${options.command}.input.json`);
  const outputPath = join(runDirectory, `${options.command}.output.json`);
  await writeFile(inputPath, JSON.stringify(options.input, null, 2), "utf-8");

  const result = await runProcess({
    executable: options.executable,
    args: [...(options.argsPrefix ?? []), options.command, inputPath, outputPath],
    cwd: runDirectory,
    timeoutMs: options.timeoutMs,
  });

  try {
    return JSON.parse(await readFile(outputPath, "utf-8")) as WorkerResponse;
  } catch (error) {
    return {
      status: "failed",
      warnings: [],
      artifacts: [],
      error: {
        code: result.timedOut ? "worker_timeout" : "worker_exit_failed",
        message: result.timedOut
          ? `Worker command timed out: ${options.command}`
          : `Worker command failed without structured output: ${options.command}`,
        details: {
          exitCode: result.exitCode,
          stderr: result.stderr,
        },
      },
    };
  }
}

interface ProcessRunOptions {
  executable: string;
  args: string[];
  cwd: string;
  timeoutMs: number;
}

interface ProcessRunResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

function runProcess(options: ProcessRunOptions): Promise<ProcessRunResult> {
  return new Promise((resolve) => {
    const child = spawn(options.executable, options.args, {
      cwd: options.cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, options.timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (exitCode) => {
      clearTimeout(timer);
      resolve({ exitCode, stdout, stderr, timedOut });
    });
  });
}
