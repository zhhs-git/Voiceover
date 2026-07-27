import type { CorrectionState } from "../../state/corrections";
import type { AnalysisState, VoiceOption } from "../../types";
import { CharacterTable } from "../CharacterTable";

interface Step3ReviewProps {
  analysis: AnalysisState;
  correctionState: CorrectionState;
  savedMessage: string | null;
  isBusy: boolean;
  isSaving: boolean;
  voices: VoiceOption[];
  onSave: () => void;
  onContinue: () => void;
  onGenderChange: (characterId: string, gender: string) => void;
  onVoiceChange: (characterId: string, voiceId: string) => void;
  onPreviewVoice: (voiceId: string) => void;
}

export function Step3Review({
  analysis,
  correctionState,
  savedMessage,
  isBusy,
  isSaving,
  voices,
  onSave,
  onContinue,
  onGenderChange,
  onVoiceChange,
  onPreviewVoice,
}: Step3ReviewProps) {
  return (
    <div className="step-workspace visible" aria-label="第 3 步：审阅">
      <header className="step-header">
        <p className="eyebrow">第 3 步，共 4 步</p>
        <h2>审阅角色</h2>
        <p className="step-desc">
          在生成音频前，校对性别、合并别名并分配音色。
        </p>
      </header>

      <div className="review-layout">
        <div className="review-table-panel">
          {savedMessage && <p className="saved-message">{savedMessage}</p>}
          <CharacterTable
            characters={analysis.characters}
            voices={voices}
            onGenderChange={onGenderChange}
            onVoiceChange={onVoiceChange}
            onPreviewVoice={onPreviewVoice}
          />
          {correctionState.dirty && (
            <p className="hint">有未保存的修改，请先保存再生成。</p>
          )}
        </div>

        <div className="review-action-panel">
          <button
            className="btn-primary"
            type="button"
            onClick={onSave}
            disabled={!correctionState.dirty || isBusy}
          >
            {isSaving ? (
              <>
                <span className="spinner" /> 保存中…
              </>
            ) : (
              "保存修改"
            )}
          </button>
          <button
            className="btn-secondary"
            type="button"
            onClick={onContinue}
            disabled={correctionState.dirty}
            title={
              correctionState.dirty
                ? "请先保存修改再继续"
                : undefined
            }
          >
            继续 → 生成
          </button>
        </div>
      </div>
    </div>
  );
}
