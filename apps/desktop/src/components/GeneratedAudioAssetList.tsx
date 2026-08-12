import { convertFileSrc } from "../lib/platform";
import type { AudioAsset, ChapterMeta } from "../types";

interface GeneratedAudioAssetListProps {
  chapters: ChapterMeta[];
  audioAssets: Record<string, AudioAsset[]>;
  activeChapterId?: string;
  onDownload: (asset: AudioAsset, chapter: ChapterMeta) => void;
  onRegenerate: (asset: AudioAsset, chapter: ChapterMeta) => void;
  disabled?: boolean;
}

function formatDuration(seconds?: number): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "";
  return `${seconds.toFixed(seconds >= 10 ? 0 : 1)} 秒`;
}

function assetFileName(asset: AudioAsset): string {
  return asset.path.split("/").pop() || `${asset.assetId}.wav`;
}

function assetTitle(asset: AudioAsset): string {
  return asset.kind === "music" ? "背景音乐" : "音效";
}

function assetAudioUrl(asset: AudioAsset): string {
  const url = convertFileSrc(asset.path);
  if (!asset.refreshKey) return url;
  return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(asset.refreshKey)}`;
}

export function GeneratedAudioAssetList({
  chapters,
  audioAssets,
  activeChapterId,
  onDownload,
  onRegenerate,
  disabled = false,
}: GeneratedAudioAssetListProps) {
  const generatedChapters = chapters
    .map((chapter) => ({
      chapter,
      assets: audioAssets[chapter.id] ?? [],
    }))
    .filter(({ chapter }) => !activeChapterId || chapter.id === activeChapterId)
    .filter(({ assets }) => assets.length > 0);

  if (generatedChapters.length === 0) return null;

  const totalAssets = generatedChapters.reduce(
    (total, { assets }) => total + assets.length,
    0,
  );

  return (
    <section className="generated-audio-assets" aria-label="背景音乐和音效">
      <div className="generated-audio-assets-header">
        <div>
          <h3>已生成的背景音乐和音效</h3>
          <p>可直接播放试听，也可以转换并下载 MP3。</p>
        </div>
        <span className="status-chip chip-done">{totalAssets} 个资源</span>
      </div>

      {generatedChapters.map(({ chapter, assets }) => (
        <div key={chapter.id} className="generated-audio-chapter">
          <h4>{chapter.title}</h4>
          {(["music", "sfx"] as const).map((kind) => {
            const group = assets.filter((asset) => asset.kind === kind);
            if (group.length === 0) return null;
            return (
              <div key={kind} className="generated-audio-group">
                <div className="generated-audio-group-title">
                  {kind === "music" ? "背景音乐" : "音效"}
                </div>
                {group.map((asset) => (
                  <div key={`${asset.kind}:${asset.assetId}`} className="generated-audio-row">
                    <div className="generated-audio-info">
                      <strong>{assetTitle(asset)}</strong>
                      <span>{assetFileName(asset)}</span>
                      <small>
                        场景 {asset.sceneId}
                        {formatDuration(asset.durationSeconds)
                          ? ` · ${formatDuration(asset.durationSeconds)}`
                          : ""}
                      </small>
                    </div>
                    <audio
                      key={`${asset.kind}:${asset.assetId}:${asset.refreshKey ?? "initial"}`}
                      controls
                      preload="metadata"
                      aria-label={`试听${assetTitle(asset)}：${assetFileName(asset)}`}
                      src={assetAudioUrl(asset)}
                    />
                    <button
                      className="btn-secondary"
                      type="button"
                      disabled={disabled}
                      onClick={() => onDownload(asset, chapter)}
                    >
                      下载 MP3
                    </button>
                    <button
                      className="btn-secondary"
                      type="button"
                      disabled={disabled}
                      onClick={() => onRegenerate(asset, chapter)}
                    >
                      重新生成
                    </button>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      ))}
    </section>
  );
}
