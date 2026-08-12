import { describe, expect, test, vi } from "vitest";
import {
  audioAssetsFromArtifacts,
  audioAssetsFromManifest,
  filterAudioAssetsToScriptPlan,
  generateChapterAudioAssets,
} from "./audioAssets";

describe("generateChapterAudioAssets", () => {
  test("sends the chapter script and output directory to the worker", async () => {
    const worker = vi.fn(async () => ({
      status: "succeeded" as const,
      artifacts: [],
      warnings: [],
      voices: [],
    }));

    const result = await generateChapterAudioAssets({
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio-assets/chapter_001",
      worker,
    });

    expect(result.status).toBe("succeeded");
    expect(worker).toHaveBeenCalledWith("generate_audio_assets", {
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio-assets/chapter_001",
      force: false,
    });
  });

  test("can target one music or sound-effect asset", async () => {
    const worker = vi.fn(async () => ({
      status: "succeeded" as const,
      artifacts: [],
      warnings: [],
      voices: [],
    }));

    await generateChapterAudioAssets({
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio-assets/chapter_001",
      force: true,
      assetId: "scene_001",
      assetKind: "music",
      worker,
    });

    expect(worker).toHaveBeenCalledWith("generate_audio_assets", {
      bookId: "book_123",
      chapterId: "chapter_001",
      scriptPath: "/tmp/book/scripts/chapter_001.json",
      outputDirectory: "/tmp/book/audio-assets/chapter_001",
      force: true,
      assetId: "scene_001",
      assetKind: "music",
    });
  });
});

describe("audio asset metadata parsing", () => {
  test("parses generated artifacts into playable assets", () => {
    expect(
      audioAssetsFromArtifacts([
        {
          kind: "stable_audio_music",
          path: "/books/book_123/audio-assets/chapter_001/music/scene_001.wav",
          metadata: {
            assetId: "scene_001",
            sceneId: "scene_001",
            model: "sm-music",
            durationSeconds: 12,
          },
        },
        {
          kind: "stable_audio_manifest",
          path: "/books/book_123/audio-assets/chapter_001/manifest.json",
        },
      ]),
    ).toEqual([
      {
        assetId: "scene_001",
        kind: "music",
        sceneId: "scene_001",
        path: "/books/book_123/audio-assets/chapter_001/music/scene_001.wav",
        model: "sm-music",
        durationSeconds: 12,
        signature: undefined,
        cacheHit: undefined,
      },
    ]);
  });

  test("restores playable assets from a manifest", () => {
    expect(
      audioAssetsFromManifest({
        version: 1,
        assets: {
          "sfx:sfx_001": {
            assetId: "sfx_001",
            kind: "sfx",
            sceneId: "scene_001",
            model: "sm-sfx",
            path: "/books/book_123/audio-assets/chapter_001/sfx/sfx_001.wav",
            durationSeconds: 2.5,
          },
        },
      }),
    ).toEqual([
      {
        assetId: "sfx_001",
        kind: "sfx",
        sceneId: "scene_001",
        path: "/books/book_123/audio-assets/chapter_001/sfx/sfx_001.wav",
        model: "sm-sfx",
        durationSeconds: 2.5,
        signature: undefined,
        cacheHit: undefined,
      },
    ]);
  });

  test("filters stale manifest assets using the current chapter audio plan", () => {
    const assets = audioAssetsFromManifest({
      version: 1,
      assets: {
        "music:scene_001": {
          assetId: "scene_001",
          kind: "music",
          sceneId: "scene_001",
          path: "/books/book_123/music/scene_001.wav",
        },
        "music:scene_1": {
          assetId: "scene_1",
          kind: "music",
          sceneId: "scene_1",
          path: "/books/book_123/music/scene_1.wav",
        },
        "sfx:sfx_1": {
          assetId: "sfx_1",
          kind: "sfx",
          sceneId: "scene_1",
          path: "/books/book_123/sfx/sfx_1.wav",
        },
      },
    });

    const filtered = filterAudioAssetsToScriptPlan(assets, {
      audioPlan: {
        scenes: [{ id: "scene_1", music: {}, sfx: [{ id: "sfx_1" }] }],
      },
    });

    expect(filtered.map((asset) => asset.assetId)).toEqual(["scene_1", "sfx_1"]);
  });
});
