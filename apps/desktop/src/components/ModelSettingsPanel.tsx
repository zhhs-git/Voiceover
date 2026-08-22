import { useEffect, useMemo, useState } from "react";

import {
  getModelSettings,
  type LlmProviderConfigUpdate,
  type ModelSettings,
  type ModelSettingsPayload,
  type TtsBackendId,
  updateModelSettings,
} from "../lib/modelSettings";

interface ModelSettingsPanelProps {
  onSaved?: (payload: ModelSettingsPayload) => void;
}

function initialDraft(payload: ModelSettingsPayload): ModelSettings {
  return { ...payload.current };
}

function initialLlmDraft(payload: ModelSettingsPayload): LlmProviderConfigUpdate {
  return {
    modelId: payload.llmConfig.modelId || payload.current.llmModelId,
    baseUrl: payload.llmConfig.baseUrl,
    apiKey: "",
    clearApiKey: false,
  };
}

export function ModelSettingsPanel({ onSaved }: ModelSettingsPanelProps) {
  const [payload, setPayload] = useState<ModelSettingsPayload | null>(null);
  const [draft, setDraft] = useState<ModelSettings | null>(null);
  const [llmDraft, setLlmDraft] = useState<LlmProviderConfigUpdate | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void getModelSettings()
      .then((next) => {
        if (cancelled) return;
        setPayload(next);
        setDraft(initialDraft(next));
        setLlmDraft(initialLlmDraft(next));
        setError(null);
      })
      .catch((loadError) => {
        if (!cancelled) setError(`无法读取模型配置：${String(loadError)}`);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedTts = useMemo(
    () => payload?.ttsOptions.find((option) => option.id === draft?.ttsBackend),
    [draft?.ttsBackend, payload?.ttsOptions],
  );
  const selectedLlmIsCatalogued = useMemo(
    () => payload?.llmOptions.some((option) => option.id === llmDraft?.modelId) ?? false,
    [llmDraft?.modelId, payload?.llmOptions],
  );

  function markDirty() {
    setSaved(false);
    setError(null);
  }

  async function save() {
    if (!draft || !llmDraft || isSaving) return;
    setIsSaving(true);
    setSaved(false);
    setError(null);
    try {
      const apiKey = llmDraft.apiKey?.trim();
      const providerUpdate = {
        modelId: llmDraft.modelId,
        baseUrl: llmDraft.baseUrl,
        ...(apiKey ? { apiKey } : {}),
        ...(llmDraft.clearApiKey ? { clearApiKey: true } : {}),
      };
      const next = await updateModelSettings(
        draft,
        llmDraft.baseUrl.trim() || apiKey || llmDraft.clearApiKey
          ? providerUpdate
          : undefined,
      );
      setPayload(next);
      setDraft(initialDraft(next));
      setLlmDraft(initialLlmDraft(next));
      setSaved(true);
      onSaved?.(next);
    } catch (saveError) {
      setError(`保存模型配置失败：${String(saveError)}`);
    } finally {
      setIsSaving(false);
    }
  }

  if (isLoading) {
    return <section className="model-settings-panel" aria-label="模型配置">正在读取模型配置…</section>;
  }

  if (!payload || !draft || !llmDraft) {
    return (
      <section className="model-settings-panel" aria-label="模型配置">
        <p className="model-settings-error" role="alert">{error || "模型配置不可用。"}</p>
      </section>
    );
  }

  return (
    <section className="model-settings-panel" aria-label="模型配置">
      <header className="model-settings-header">
        <div>
          <h2>模型配置</h2>
          <p>LLM 连接配置保存在本项目 .env；API Key 仅可写入，页面不会显示已保存内容。</p>
        </div>
      </header>

      <div className="model-settings-fields">
        <label className="model-settings-field">
          <span>分析模型</span>
          <select
            aria-label="分析模型"
            value={llmDraft.modelId}
            onChange={(event) => {
              markDirty();
              setDraft((current) => current && {
                ...current,
                llmModelId: event.target.value,
              });
              setLlmDraft((current) => current && {
                ...current,
                modelId: event.target.value,
              });
            }}
            disabled={isSaving}
          >
            {!selectedLlmIsCatalogued && llmDraft.modelId && (
              <option value={llmDraft.modelId}>自定义：{llmDraft.modelId}</option>
            )}
            {payload.llmOptions.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available}>
                {option.displayName}{option.available ? "" : "（不可用）"}
              </option>
            ))}
          </select>
        </label>

        <label className="model-settings-field">
          <span>分析模型 ID</span>
          <input
            aria-label="分析模型 ID"
            value={llmDraft.modelId}
            onChange={(event) => {
              markDirty();
              const modelId = event.target.value;
              setDraft((current) => current && { ...current, llmModelId: modelId });
              setLlmDraft((current) => current && { ...current, modelId });
            }}
            disabled={isSaving}
            autoComplete="off"
            spellCheck={false}
          />
        </label>

        <label className="model-settings-field model-settings-field-wide">
          <span>LLM 服务 URL</span>
          <input
            aria-label="LLM 服务 URL"
            type="url"
            value={llmDraft.baseUrl}
            onChange={(event) => {
              markDirty();
              setLlmDraft((current) => current && {
                ...current,
                baseUrl: event.target.value,
              });
            }}
            disabled={isSaving}
            autoComplete="url"
            spellCheck={false}
            placeholder="https://api.example.com/v1"
          />
        </label>

        <div className="model-settings-field model-settings-field-wide">
          <label htmlFor="llm-api-key">LLM API Key</label>
          <input
            id="llm-api-key"
            aria-label="LLM API Key"
            type="password"
            value={llmDraft.apiKey || ""}
            onChange={(event) => {
              markDirty();
              setLlmDraft((current) => current && {
                ...current,
                apiKey: event.target.value,
                clearApiKey: false,
              });
            }}
            disabled={isSaving || llmDraft.clearApiKey}
            autoComplete="new-password"
            spellCheck={false}
            placeholder={payload.llmConfig.apiKeyConfigured ? "已配置；留空则保持不变" : "输入后仅写入本机 .env"}
          />
          <small className="model-settings-key-status" aria-live="polite">
            API Key：{payload.llmConfig.apiKeyConfigured ? "已配置" : "未配置"}
          </small>
          <label className="model-settings-checkbox">
            <input
              aria-label="清除已保存的 API Key"
              type="checkbox"
              checked={Boolean(llmDraft.clearApiKey)}
              onChange={(event) => {
                markDirty();
                setLlmDraft((current) => current && {
                  ...current,
                  apiKey: "",
                  clearApiKey: event.target.checked,
                });
              }}
              disabled={isSaving || !payload.llmConfig.apiKeyConfigured}
            />
            <span>清除已保存的 API Key</span>
          </label>
        </div>

        <label className="model-settings-field">
          <span>配音模型</span>
          <select
            aria-label="配音模型"
            value={draft.ttsBackend}
            onChange={(event) => {
              const backend = event.target.value as TtsBackendId;
              const option = payload.ttsOptions.find((item) => item.id === backend);
              if (!option || !option.available) return;
              markDirty();
              setDraft((current) => current && {
                ...current,
                ttsBackend: backend,
                ttsModelId: option.modelId,
              });
            }}
            disabled={isSaving}
          >
            {payload.ttsOptions.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available}>
                {option.displayName}{option.available ? "" : "（不可用）"}
              </option>
            ))}
          </select>
          {selectedTts && !selectedTts.available && (
            <small className="model-settings-unavailable">{selectedTts.reason}</small>
          )}
        </label>
      </div>

      {payload.ttsOptions.some((option) => !option.available) && (
        <div className="model-settings-diagnostics" aria-label="模型可用性诊断">
          {payload.ttsOptions
            .filter((option) => !option.available)
            .map((option) => (
              <p className="model-settings-unavailable" key={option.id}>
                <strong>{option.displayName}</strong>：{option.reason}
              </p>
            ))}
        </div>
      )}

      <div className="model-settings-actions">
        <button
          type="button"
          className="btn-primary model-settings-save"
          onClick={() => void save()}
          disabled={isSaving}
        >
          {isSaving ? "正在保存…" : "保存模型配置"}
        </button>
        {saved && <span className="model-settings-saved">已保存</span>}
      </div>
      {error && <p className="model-settings-error" role="alert">{error}</p>}
    </section>
  );
}
