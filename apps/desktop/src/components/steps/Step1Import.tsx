import type { BookState, RightsResult } from "../../types";

interface Step1ImportProps {
  book: BookState | null;
  rights: RightsResult | null;
  rightsAttested: boolean;
  isBusy: boolean;
  isImporting: boolean;
  onImport: () => void;
  onAttest: (checked: boolean) => void;
  onContinue: () => void;
}

export function Step1Import({
  book,
  rights,
  rightsAttested,
  isBusy,
  isImporting,
  onImport,
  onAttest,
  onContinue,
}: Step1ImportProps) {
  const blocked = rights?.classification === "blocked";
  const needsAttestation = rights?.requiresAttestation && !rightsAttested;
  const rightsLabel = rights
    ? {
        allowed: "可转换",
        restricted: "受限",
        blocked: "禁止",
        unknown: "未知",
      }[rights.classification] ?? rights.classification
    : "";

  return (
    <div className="step-workspace visible" aria-label="第 1 步：导入">
      <header className="step-header">
        <p className="eyebrow">第 1 步，共 4 步</p>
        <h2>导入书籍</h2>
        <p className="step-desc">
          选择 EPUB、PDF 或 TXT 文件，提取章节并检查版权。
        </p>
      </header>

      <div className="import-area">
        <button
          className="btn-primary btn-import"
          type="button"
          onClick={onImport}
          disabled={isBusy}
        >
          {isImporting ? (
            <>
              <span className="spinner" /> 导入中…
            </>
          ) : (
            <>
              <span className="import-icon">📂</span> 选择文件
            </>
          )}
        </button>
        {isImporting && <p className="import-hint">正在提取章节…</p>}
      </div>

      {book && (
        <div className="import-result">
          <div className="result-row">
            <span className="result-label">书名</span>
            <span className="result-value">{book.title}</span>
          </div>
          <div className="result-row">
            <span className="result-label">章节数</span>
            <span className="result-value">{book.chapters.length}</span>
          </div>

          {rights && (
            <div className="result-row">
              <span className="result-label">版权</span>
              <span className={`rights-strip rights-${rights.classification}`}>
                {rightsLabel}
              </span>
            </div>
          )}

          {rights?.requiresAttestation && (
            <label className="attestation">
              <input
                type="checkbox"
                checked={rightsAttested}
                onChange={(e) => onAttest(e.target.checked)}
              />
              <span>我已获得转换本书的授权</span>
            </label>
          )}

          <button
            className="btn-primary"
            type="button"
            onClick={onContinue}
            disabled={blocked || needsAttestation}
          >
            继续 → 分析
          </button>
        </div>
      )}
    </div>
  );
}
