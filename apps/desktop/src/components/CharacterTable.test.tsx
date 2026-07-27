import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { CharacterTable } from "./CharacterTable";

const sampleCharacters = [
  { id: "elizabeth", canonicalName: "Elizabeth", aliases: ["Lizzy"], gender: "female", voiceId: "female_adult_01", confidence: 0.92 },
  { id: "darcy", canonicalName: "Darcy", aliases: [], gender: "male", voiceId: "male_adult_01", confidence: 0.78 },
  { id: "unknown_speaker", canonicalName: "Unknown Speaker", aliases: [], gender: "unknown", voiceId: "neutral_dialogue_01", confidence: 0.35 },
];

const VOICE_OPTIONS = [
  { id: "narrator_default", displayName: "Default Narrator" },
  { id: "female_adult_01", displayName: "Female Adult 01" },
  { id: "male_adult_01", displayName: "Male Adult 01" },
  { id: "neutral_dialogue_01", displayName: "Neutral Dialogue 01" },
];

describe("CharacterTable", () => {
  test("renders all characters with name, gender, voice, and confidence", () => {
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
      />
    );

    expect(screen.getByText("Elizabeth")).toBeInTheDocument();
    expect(screen.getByText("Darcy")).toBeInTheDocument();
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
  });

  test("low-confidence character shows warning indicator", () => {
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
      />
    );

    const unknownRow = screen.getByText("35%").closest("tr");
    expect(unknownRow).toBeTruthy();
    expect(unknownRow!.className).toContain("low-confidence");
    expect(screen.getByLabelText("置信度较低")).toBeInTheDocument();
  });

  test("calls onGenderChange when gender dropdown changes", () => {
    const onGenderChange = vi.fn();
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={onGenderChange}
        onVoiceChange={() => {}}
      />
    );

    const genderSelect = screen.getAllByLabelText("性别")[0] as HTMLSelectElement;
    genderSelect.value = "male";
    genderSelect.dispatchEvent(new Event("change", { bubbles: true }));
    expect(onGenderChange).toHaveBeenCalledWith("elizabeth", "male");
  });

  test("calls onVoiceChange when voice dropdown changes", () => {
    const onVoiceChange = vi.fn();
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={onVoiceChange}
      />
    );

    const voiceSelect = screen.getAllByLabelText("音色")[0] as HTMLSelectElement;
    voiceSelect.value = "male_adult_01";
    voiceSelect.dispatchEvent(new Event("change", { bubbles: true }));
    expect(onVoiceChange).toHaveBeenCalledWith("elizabeth", "male_adult_01");
  });

  test("calls onPreviewVoice for the selected character voice", () => {
    const onPreviewVoice = vi.fn();
    render(
      <CharacterTable
        characters={sampleCharacters}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
        onPreviewVoice={onPreviewVoice}
      />
    );

    screen.getAllByRole("button", { name: /试听音色/i })[0].click();
    expect(onPreviewVoice).toHaveBeenCalledWith("female_adult_01");
  });

  test("shows empty state message when no characters", () => {
    render(
      <CharacterTable
        characters={[]}
        voices={VOICE_OPTIONS}
        onGenderChange={() => {}}
        onVoiceChange={() => {}}
      />
    );

    expect(screen.getByText("暂未检测到角色，请先运行分析。")).toBeInTheDocument();
  });
});
