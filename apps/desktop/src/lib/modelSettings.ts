import { invoke } from "./platform";

export type TtsBackendId = "mimo" | "voxcpm2";

export interface ModelSettings {
  llmModelId: string;
  ttsBackend: TtsBackendId;
  ttsModelId: string;
}

/** Safe, read-only projection of the active OpenAI-compatible provider. */
export interface LlmProviderConfig {
  modelId: string;
  baseUrl: string;
  apiKeyConfigured: boolean;
}

/** Write-only provider fields accepted by model_settings_update. */
export interface LlmProviderConfigUpdate {
  modelId: string;
  baseUrl: string;
  apiKey?: string;
  clearApiKey?: boolean;
}

export interface LlmModelOption {
  id: string;
  provider: string;
  displayName: string;
  family: string;
  available: boolean;
}

export interface TtsModelOption {
  id: TtsBackendId;
  modelId: string;
  displayName: string;
  available: boolean;
  reason: string;
}

export interface ModelSettingsPayload {
  version: number;
  current: ModelSettings;
  llmConfig: LlmProviderConfig;
  llmOptions: LlmModelOption[];
  ttsOptions: TtsModelOption[];
}

export async function getModelSettings(): Promise<ModelSettingsPayload> {
  return invoke<ModelSettingsPayload>("model_settings_get");
}

export async function updateModelSettings(
  settings: ModelSettings,
  llmConfig?: LlmProviderConfigUpdate,
): Promise<ModelSettingsPayload> {
  return invoke<ModelSettingsPayload>("model_settings_update", {
    ...settings,
    ...(llmConfig ? { llmConfig } : {}),
  });
}

export function llmDisplayName(payload: ModelSettingsPayload): string {
  const option = payload.llmOptions.find(
    (item) => item.id === payload.current.llmModelId,
  );
  return option?.displayName || payload.current.llmModelId;
}

export function backendScopedVoiceProfileDirectory(
  workDir: string,
  backend: TtsBackendId,
): string {
  return backend === "voxcpm2"
    ? `${workDir}/voice-profiles/voxcpm2`
    : `${workDir}/voice-profiles`;
}

export function backendScopedVoicePreviewDirectory(
  workDir: string,
  backend: TtsBackendId,
): string {
  return backend === "voxcpm2"
    ? `${workDir}/voice-previews/voxcpm2`
    : `${workDir}/voice-previews`;
}
