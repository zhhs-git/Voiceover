import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import type { AudioAsset, ChapterMeta } from "../types";
import { GeneratedAudioAssetList } from "./GeneratedAudioAssetList";

vi.mock("../lib/platform", () => ({
  convertFileSrc: (path: string) => `/api/files?path=${encodeURIComponent(path)}`,
}));

const chapter: ChapterMeta = {
  id: "chapter_001",
  title: "第一章",
  textLength: 100,
  textPath: "/books/book_123/chapters/chapter_001.txt",
};

const secondChapter: ChapterMeta = {
  id: "chapter_002",
  title: "第二章",
  textLength: 120,
  textPath: "/books/book_123/chapters/chapter_002.txt",
};

const assets: AudioAsset[] = [
  {
    assetId: "scene_001",
    kind: "music",
    sceneId: "scene_001",
    path: "/books/book_123/audio-assets/chapter_001/music/scene_001.wav",
    durationSeconds: 12,
  },
  {
    assetId: "sfx_001",
    kind: "sfx",
    sceneId: "scene_001",
    path: "/books/book_123/audio-assets/chapter_001/sfx/sfx_001.wav",
    durationSeconds: 2.5,
  },
];

describe("GeneratedAudioAssetList", () => {
  test("renders playable music and sound-effect rows", () => {
    const onDownload = vi.fn();
    const onRegenerate = vi.fn();
    const { container } = render(
      <GeneratedAudioAssetList
        chapters={[chapter]}
        audioAssets={{ chapter_001: assets }}
        onDownload={onDownload}
        onRegenerate={onRegenerate}
      />,
    );

    expect(screen.getByText("已生成的背景音乐和音效")).toBeInTheDocument();
    expect(screen.getByText("scene_001.wav")).toBeInTheDocument();
    expect(screen.getByText("sfx_001.wav")).toBeInTheDocument();
    expect(container.querySelectorAll("audio")).toHaveLength(2);
    expect(container.querySelector("audio")).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio-assets%2Fchapter_001%2Fmusic%2Fscene_001.wav",
    );

    screen.getAllByRole("button", { name: "下载 MP3" })[0].click();
    expect(onDownload).toHaveBeenCalledWith(assets[0], chapter);
    screen.getAllByRole("button", { name: "重新生成" })[0].click();
    expect(onRegenerate).toHaveBeenCalledWith(assets[0], chapter);
  });

  test("renders nothing when a book has no generated assets", () => {
    const { container } = render(
      <GeneratedAudioAssetList
        chapters={[chapter]}
        audioAssets={{}}
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  test("refreshes the player URL when an asset is regenerated in place", () => {
    const { container, rerender } = render(
      <GeneratedAudioAssetList
        chapters={[chapter]}
        audioAssets={{ chapter_001: [assets[0]] }}
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );
    const firstPlayer = container.querySelector("audio");

    rerender(
      <GeneratedAudioAssetList
        chapters={[chapter]}
        audioAssets={{
          chapter_001: [{ ...assets[0], refreshKey: "regenerated-2" }],
        }}
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    const refreshedPlayer = container.querySelector("audio");
    expect(refreshedPlayer).not.toBe(firstPlayer);
    expect(refreshedPlayer).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio-assets%2Fchapter_001%2Fmusic%2Fscene_001.wav&v=regenerated-2",
    );
  });

  test("only renders assets for the selected listening chapter", () => {
    const secondAsset: AudioAsset = {
      ...assets[0],
      assetId: "scene_002",
      sceneId: "scene_002",
      path: "/books/book_123/audio-assets/chapter_002/music/scene_002.wav",
    };
    const { container, rerender } = render(
      <GeneratedAudioAssetList
        chapters={[chapter, secondChapter]}
        audioAssets={{ chapter_001: [assets[0]], chapter_002: [secondAsset] }}
        activeChapterId="chapter_001"
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    expect(container.querySelectorAll("audio")).toHaveLength(1);
    expect(screen.getByText("scene_001.wav")).toBeInTheDocument();
    expect(screen.queryByText("scene_002.wav")).not.toBeInTheDocument();

    rerender(
      <GeneratedAudioAssetList
        chapters={[chapter, secondChapter]}
        audioAssets={{ chapter_001: [assets[0]], chapter_002: [secondAsset] }}
        activeChapterId="chapter_002"
        onDownload={vi.fn()}
        onRegenerate={vi.fn()}
      />,
    );

    expect(container.querySelector("audio")).toHaveAttribute(
      "src",
      "/api/files?path=%2Fbooks%2Fbook_123%2Faudio-assets%2Fchapter_002%2Fmusic%2Fscene_002.wav",
    );
    expect(screen.queryByText("scene_001.wav")).not.toBeInTheDocument();
    expect(screen.getByText("scene_002.wav")).toBeInTheDocument();
  });
});
