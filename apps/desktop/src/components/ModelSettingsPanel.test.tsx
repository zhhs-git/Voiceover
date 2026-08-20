import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

const modelSettings = vi.hoisted(() => ({
  getModelSettings: vi.fn(),
  updateModelSettings: vi.fn(),
}));

vi.mock("../lib/modelSettings", () => ({
  getModelSettings: modelSettings.getModelSettings,
  updateModelSettings: modelSettings.updateModelSettings,
}));

import { ModelSettingsPanel } from "./ModelSettingsPanel";

const payload = {
  version: 1,
  current: {
    llmModelId: "openai/gpt-5.6-terra",
    ttsBackend: "mimo" as const,
    ttsModelId: "mimo-v2.5-tts-voiceclone",
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
      id: "mimo" as const,
      modelId: "mimo-v2.5-tts-voiceclone",
      displayName: "MiMo Voice Clone",
      available: true,
      reason: "使用现有 MiMo voice-clone 流程。",
    },
    {
      id: "voxcpm2" as const,
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

  test("loads safe options and saves the selected model tuple", async () => {
    const onSaved = vi.fn();
    modelSettings.getModelSettings.mockResolvedValue(payload);
    modelSettings.updateModelSettings.mockResolvedValue(payload);

    render(<ModelSettingsPanel onSaved={onSaved} />);

    await waitFor(() => {
      expect(screen.getByLabelText("分析模型")).toHaveValue("openai/gpt-5.6-terra");
    });
    expect(screen.getByLabelText("配音模型")).toHaveValue("mimo");
    expect(screen.getByRole("option", { name: "VoxCPM2（本地）（不可用）" })).toBeDisabled();
    expect(screen.queryByText(/api key|secret|token/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "保存模型配置" }));

    await waitFor(() => {
      expect(modelSettings.updateModelSettings).toHaveBeenCalledWith(payload.current);
    });
    expect(onSaved).toHaveBeenCalledWith(payload);
    expect(screen.getByText("已保存")).toBeInTheDocument();
  });

  test("renders an actionable diagnostic for an unavailable selected backend", async () => {
    modelSettings.getModelSettings.mockResolvedValue({
      ...payload,
      current: {
        ...payload.current,
        ttsBackend: "voxcpm2" as const,
        ttsModelId: "VoxCPM2",
      },
    });

    render(<ModelSettingsPanel />);

    expect(await screen.findByText("缺少独立 Python 环境。")).toBeInTheDocument();
  });
});
