import { workerCall } from "./workerCall";

import type { WorkerResponse } from "@audiobook-generator/shared";
import type { AudioAsset } from "../types";

type WorkerCall = (
  command: string,
  input: Record<string, unknown>,
) => Promise<WorkerResponse & Record<string, unknown>>;

interface GenerateChapterAudioAssetsInput {
  bookId: string;
  chapterId: string;
  scriptPath: string;
  outputDirectory: string;
  mixedOutputPath?: string;
  force?: boolean;
  assetId?: string;
  assetKind?: AudioAsset["kind"];
  worker?: WorkerCall;
}

export async function generateChapterAudioAssets({
  bookId,
  chapterId,
  scriptPath,
  outputDirectory,
  mixedOutputPath,
  force = false,
  assetId,
  assetKind,
  worker = workerCall,
}: GenerateChapterAudioAssetsInput): Promise<Record<string, unknown>> {
  const input: Record<string, unknown> = {
    bookId,
    chapterId,
    scriptPath,
    outputDirectory,
    force,
  };
  if (mixedOutputPath) input.mixedOutputPath = mixedOutputPath;
  if (assetId) input.assetId = assetId;
  if (assetKind) input.assetKind = assetKind;
  return worker("generate_audio_assets", input);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeAudioAsset(
  value: Record<string, unknown>,
  fallbackAssetId?: string,
): AudioAsset | null {
  const rawKind = typeof value.kind === "string" ? value.kind : "";
  const kind = rawKind.startsWith("stable_audio_")
    ? rawKind.slice("stable_audio_".length)
    : rawKind;
  if (kind !== "music" && kind !== "sfx") return null;

  const path = typeof value.path === "string" ? value.path : "";
  if (!path) return null;

  const assetId =
    typeof value.assetId === "string" && value.assetId
      ? value.assetId
      : fallbackAssetId || path.split("/").pop() || "audio_asset";
  const sceneId =
    typeof value.sceneId === "string" && value.sceneId
      ? value.sceneId
      : assetId;

  return {
    assetId,
    kind,
    sceneId,
    path,
    model: typeof value.model === "string" ? value.model : undefined,
    durationSeconds: optionalNumber(value.durationSeconds),
    signature: typeof value.signature === "string" ? value.signature : undefined,
    cacheHit: typeof value.cacheHit === "boolean" ? value.cacheHit : undefined,
  };
}

export function audioAssetsFromArtifacts(value: unknown): AudioAsset[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((artifact) => {
    if (!isRecord(artifact)) return [];
    const metadata = isRecord(artifact.metadata) ? artifact.metadata : {};
    const normalized = normalizeAudioAsset({
      ...metadata,
      kind: artifact.kind,
      path: artifact.path,
    });
    return normalized ? [normalized] : [];
  });
}

export function audioAssetsFromManifest(value: unknown): AudioAsset[] {
  if (!isRecord(value) || !isRecord(value.assets)) return [];
  return Object.entries(value.assets).flatMap(([manifestKey, asset]) => {
    if (!isRecord(asset)) return [];
    const fallbackAssetId = manifestKey.includes(":")
      ? manifestKey.slice(manifestKey.indexOf(":") + 1)
      : manifestKey;
    const normalized = normalizeAudioAsset(asset, fallbackAssetId);
    return normalized ? [normalized] : [];
  });
}

function audioAssetKeysFromScript(value: unknown): Set<string> | null {
  if (!isRecord(value) || !isRecord(value.audioPlan)) return null;
  const scenes = value.audioPlan.scenes;
  if (!Array.isArray(scenes)) return null;

  const keys = new Set<string>();
  for (const scene of scenes) {
    if (!isRecord(scene) || typeof scene.id !== "string" || !scene.id) continue;
    if (isRecord(scene.music)) keys.add(`music:${scene.id}`);
    if (Array.isArray(scene.sfx)) {
      for (const effect of scene.sfx) {
        if (isRecord(effect) && typeof effect.id === "string" && effect.id) {
          keys.add(`sfx:${effect.id}`);
        }
      }
    }
  }
  return keys;
}

export function filterAudioAssetsToScriptPlan(
  assets: AudioAsset[],
  script: unknown,
): AudioAsset[] {
  const allowedKeys = audioAssetKeysFromScript(script);
  if (!allowedKeys) return assets;
  return assets.filter((asset) => allowedKeys.has(`${asset.kind}:${asset.assetId}`));
}
