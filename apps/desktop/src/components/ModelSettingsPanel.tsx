import { useEffect, useMemo, useState } from "react";

import {
  getModelSettings,
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

export function ModelSettingsPanel({ onSaved }: ModelSettingsPanelProps) {
  const [payload, setPayload] = useState<ModelSettingsPayload | null>(null);
  const [draft, setDraft] = useState<ModelSettings | null>(null);
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

  async function save() {
    if (!draft || isSaving) return;
    setIsSaving(true);
    setSaved(false);
    setError(null);
    try {
      const next = await updateModelSettings(draft);
      setPayload(next);
      setDraft(initialDraft(next));
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

  if (!payload || !draft) {
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
          <p>当前设置</p>
        </div>
      </header>

      <div className="model-settings-fields">
        <label className="model-settings-field">
          <span>分析模型</span>
          <select
            aria-label="分析模型"
            value={draft.llmModelId}
            onChange={(event) => {
              setSaved(false);
              setDraft((current) => current && {
                ...current,
                llmModelId: event.target.value,
              });
            }}
            disabled={isSaving}
          >
            {payload.llmOptions.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available}>
                {option.displayName}
              </option>
            ))}
          </select>
        </label>

        <label className="model-settings-field">
          <span>配音模型</span>
          <select
            aria-label="配音模型"
            value={draft.ttsBackend}
            onChange={(event) => {
              const backend = event.target.value as TtsBackendId;
              const option = payload.ttsOptions.find((item) => item.id === backend);
              if (!option || !option.available) return;
              setSaved(false);
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
