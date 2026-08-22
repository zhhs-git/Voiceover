import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import type { ModelSettingsPayload } from "../lib/modelSettings";

const modelSettings = vi.hoisted(() => ({
  getModelSettings: vi.fn(),
  updateModelSettings: vi.fn(),
}));

vi.mock("../lib/modelSettings", () => ({
  getModelSettings: modelSettings.getModelSettings,
  updateModelSettings: modelSettings.updateModelSettings,
}));

import { ModelSettingsPanel } from "./ModelSettingsPanel";

const payload: ModelSettingsPayload = {
  version: 2,
  current: {
    llmModelId: "openai/gpt-5.6-terra",
    ttsBackend: "mimo",
    ttsModelId: "mimo-v2.5-tts-voiceclone",
  },
  llmConfig: {
    modelId: "openai/gpt-5.6-terra",
    baseUrl: "https://gateway.example/v1",
    apiKeyConfigured: true,
  },
  llmOptions: [{
    id: "openai/gpt-5.6-terra",
    provider: "openai",
    displayName: "GPT-5.6 Terra",
    family: "default",
    available: true,
  }],
  ttsOptions: [
    {
      id: "mimo",
      modelId: "mimo-v2.5-tts-voiceclone",
      displayName: "MiMo Voice Clone",
      available: true,
      reason: "使用现有 MiMo voice-clone 流程。",
    },
    {
      id: "voxcpm2",
      modelId: "VoxCPM2",
      displayName: "VoxCPM2（本地）",
      available: false,
      reason: "缺少独立 Python 环境。",
    },
  ],
};

describe("ModelSettingsPanel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  test("loads the provider safely and saves the write-only LLM fields", async () => {
    const onSaved = vi.fn();
    modelSettings.getModelSettings.mockResolvedValue(payload);
    modelSettings.updateModelSettings.mockResolvedValue(payload);

    render(<ModelSettingsPanel onSaved={onSaved} />);

    await waitFor(() => {
      expect(screen.getByLabelText("分析模型")).toHaveValue("openai/gpt-5.6-terra");
    });
    expect(screen.getByLabelText("分析模型 ID")).toHaveValue("openai/gpt-5.6-terra");
    expect(screen.getByLabelText("LLM 服务 URL")).toHaveValue("https://gateway.example/v1");
    expect(screen.getByLabelText("LLM API Key")).toHaveValue("");
    expect(screen.getByText("API Key：已配置")).toBeInTheDocument();
    expect(screen.getByLabelText("配音模型")).toHaveValue("mimo");
    expect(screen.getByRole("option", { name: "VoxCPM2（本地）（不可用）" })).toBeDisabled();

    fireEvent.change(screen.getByLabelText("LLM 服务 URL"), {
      target: { value: "https://new-gateway.example/v1" },
    });
    fireEvent.change(screen.getByLabelText("LLM API Key"), {
      target: { value: "new-test-secret" },
    });
    fireEvent.click(screen.getByRole("button", { name: "保存模型配置" }));

    await waitFor(() => {
      expect(modelSettings.updateModelSettings).toHaveBeenCalledWith(payload.current, {
        modelId: "openai/gpt-5.6-terra",
        baseUrl: "https://new-gateway.example/v1",
        apiKey: "new-test-secret",
      });
    });
    expect(onSaved).toHaveBeenCalledWith(payload);
    expect(screen.getByText("已保存")).toBeInTheDocument();
  });

  test("sends an explicit clear request without reading or echoing the API key", async () => {
    modelSettings.getModelSettings.mockResolvedValue(payload);
    modelSettings.updateModelSettings.mockResolvedValue({
      ...payload,
      llmConfig: { ...payload.llmConfig, apiKeyConfigured: false },
    });

    render(<ModelSettingsPanel />);

    const keyField = await screen.findByLabelText("LLM API Key");
    expect(keyField).toHaveValue("");
    fireEvent.click(screen.getByLabelText("清除已保存的 API Key"));
    expect(keyField).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "保存模型配置" }));

    await waitFor(() => {
      expect(modelSettings.updateModelSettings).toHaveBeenCalledWith(payload.current, {
        modelId: "openai/gpt-5.6-terra",
        baseUrl: "https://gateway.example/v1",
        clearApiKey: true,
      });
    });
    expect(screen.getByText("API Key：未配置")).toBeInTheDocument();
  });

  test("keeps the legacy tuple-only save path when no provider endpoint exists", async () => {
    const withoutProvider: ModelSettingsPayload = {
      ...payload,
      llmConfig: {
        modelId: payload.current.llmModelId,
        baseUrl: "",
        apiKeyConfigured: false,
      },
    };
    modelSettings.getModelSettings.mockResolvedValue(withoutProvider);
    modelSettings.updateModelSettings.mockResolvedValue(withoutProvider);

    render(<ModelSettingsPanel />);

    await screen.findByLabelText("LLM 服务 URL");
    fireEvent.click(screen.getByRole("button", { name: "保存模型配置" }));

    await waitFor(() => {
      expect(modelSettings.updateModelSettings).toHaveBeenCalledWith(
        withoutProvider.current,
        undefined,
      );
    });
  });

  test("renders an actionable diagnostic for an unavailable selected backend", async () => {
    modelSettings.getModelSettings.mockResolvedValue({
      ...payload,
      current: {
        ...payload.current,
        ttsBackend: "voxcpm2",
        ttsModelId: "VoxCPM2",
      },
    });

    render(<ModelSettingsPanel />);

    expect(await screen.findByText("缺少独立 Python 环境。")).toBeInTheDocument();
  });
});
