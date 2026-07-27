import { invoke } from "./platform";
import { WorkerResponseSchema, type WorkerResponse } from "@audiobook-generator/shared";

export async function workerCall(
  command: string,
  input: Record<string, unknown>,
): Promise<WorkerResponse & Record<string, unknown>> {
  const raw = await invoke<string>("run_worker", {
    command,
    inputJson: JSON.stringify(input),
  });
  const parsed: unknown = JSON.parse(raw);
  return WorkerResponseSchema.parse(parsed) as WorkerResponse & Record<string, unknown>;
}
