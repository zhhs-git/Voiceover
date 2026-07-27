import type { CharacterMeta, VoiceOption } from "../types";
import { localizeVoiceDisplayName } from "../lib/voiceOptions";

interface CharacterTableProps {
  characters: CharacterMeta[];
  voices: VoiceOption[];
  onGenderChange: (characterId: string, gender: string) => void;
  onVoiceChange: (characterId: string, voiceId: string) => void;
  onPreviewVoice?: (voiceId: string) => void;
}

function confidenceColor(confidence: number): string {
  if (confidence >= 0.8) return "var(--success)";
  if (confidence >= 0.5) return "var(--warning)";
  return "var(--danger)";
}

export function CharacterTable({
  characters,
  voices,
  onGenderChange,
  onVoiceChange,
  onPreviewVoice,
}: CharacterTableProps) {
  if (characters.length === 0) {
    return <p>暂未检测到角色，请先运行分析。</p>;
  }

  return (
    <table className="character-table">
      <thead>
        <tr>
          <th>角色名</th>
          <th>别名</th>
          <th>置信度</th>
          <th>性别</th>
          <th>音色</th>
        </tr>
      </thead>
      <tbody>
        {characters.map((c) => (
          <tr key={c.id} className={c.confidence < 0.5 ? "low-confidence" : ""}>
            <td>
              {c.canonicalName}
              {c.identityStatus === "provisional" && (
                <span aria-label="角色身份待确认" className="warning-icon" title="角色身份待确认">
                  ?
                </span>
              )}
              {c.confidence < 0.5 && (
                <span aria-label="置信度较低" className="warning-icon">
                  ⚠
                </span>
              )}
            </td>
            <td>{c.aliases.length > 0 ? c.aliases.join(", ") : "—"}</td>
            <td>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.round(c.confidence * 100)}%`,
                    backgroundColor: confidenceColor(c.confidence),
                  }}
                />
                <span className="confidence-label">{Math.round(c.confidence * 100)}%</span>
              </div>
            </td>
            <td>
              <select
                className="dark-select"
                aria-label="性别"
                value={c.gender}
                onChange={(e) => onGenderChange(c.id, e.target.value)}
              >
                <option value="unknown">未知</option>
                <option value="female">女性</option>
                <option value="male">男性</option>
                <option value="neutral">中性</option>
              </select>
            </td>
            <td>
              <div className="voice-control">
                <select
                  className="dark-select"
                  aria-label="音色"
                  value={c.voiceId}
                  onChange={(e) => onVoiceChange(c.id, e.target.value)}
                >
                  {voices.map((v) => (
                    <option key={v.id} value={v.id} disabled={v.available === false}>
                      {localizeVoiceDisplayName(v.displayName, v.id)}{v.available === false ? "（不可用）" : ""}
                    </option>
                  ))}
                </select>
                {onPreviewVoice && (
                  <button
                    className="icon-btn"
                    type="button"
                    aria-label={`试听音色 ${c.voiceId}`}
                    title="试听音色"
                    onClick={() => onPreviewVoice(c.voiceId)}
                  >
                    ▶
                  </button>
                )}
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
