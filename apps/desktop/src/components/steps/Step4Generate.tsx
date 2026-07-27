import type { AnalysisState, BookState, ProgressDetail } from "../../types";

interface Step4GenerateProps {
  book: BookState;
  analysis: AnalysisState;
  selectedChapters: Set<string>;
  correctionDirty: boolean;
  chapterAudioPaths: Record<string, string>;
  isBusy: boolean;
  isGenerating: boolean;
  analyzeProgress: string;
  progressDetail: ProgressDetail[];
  progress: number;
  onGenerate: () => void;
  onContinue: () => void;
}

export function Step4Generate({
  book,
  analysis,
  selectedChapters,
  correctionDirty,
  chapterAudioPaths,
  isBusy,
  isGenerating,
  analyzeProgress,
  progressDetail,
  progress,
  onGenerate,
  onContinue,
}: Step4GenerateProps) {
  const chaptersReady = book.chapters.filter(
    (c) => selectedChapters.has(c.id) && analysis.scriptPaths[c.id],
  );
  const audioCount = Object.keys(chapterAudioPaths).length;

  return (
    <div className="step-workspace visible" aria-label="第 4 步：生成">
      <header className="step-header">
        <p className="eyebrow">第 4 步，共 4 步</p>
        <h2>生成音频</h2>
        <p className="step-desc">
          使用 MiMo V2.5 TTS 音色设计合成选中的章节。
        </p>
      </header>

      <div className="generate-layout">
        <div className="generate-info">
          <div className="result-row">
            <span className="result-label">待生成</span>
            <span className="result-value">{chaptersReady.length} 章</span>
          </div>
          <div className="result-row">
            <span className="result-label">后端</span>
            <span className="result-value">MiMo V2.5 TTS 音色设计</span>
          </div>
          {correctionDirty && (
            <div className="result-row">
              <span className="result-label warn">⚠</span>
              <span className="result-value warn">有未保存的修改，请返回角色审阅。</span>
            </div>
          )}
        </div>

        <div className="generate-action">
          {analyzeProgress && (
            <p className="analyze-progress">{analyzeProgress}</p>
          )}
          {progressDetail.length > 0 && isGenerating && (
            <div className="progress-detail">
              {progressDetail.map((d) => (
                <span key={d.label} className="progress-detail-item">
                  <strong>{d.label}</strong> {d.value}
                </span>
              ))}
            </div>
          )}
          <progress value={progress} max="100" aria-label="生成进度" />
          <button
            className="btn-primary"
            type="button"
            onClick={onGenerate}
            disabled={isBusy || chaptersReady.length === 0}
          >
            {isGenerating ? (
              <>
                <span className="spinner" /> 生成中…
              </>
            ) : (
              `生成 ${chaptersReady.length} 章`
            )}
          </button>
          {audioCount > 0 && !isGenerating && (
            <button className="btn-secondary" type="button" onClick={onContinue}>
              试听结果 →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
