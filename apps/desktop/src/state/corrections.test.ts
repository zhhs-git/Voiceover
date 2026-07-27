import { describe, expect, test } from "vitest";
import {
  createCorrectionsStore,
  type CorrectionState,
  type AliasMerge,
  type GenderOverride,
  type VoiceOverride,
} from "./corrections";

describe("corrections store", () => {
  test("starts with empty corrections and clean state", () => {
    const store = createCorrectionsStore();
    expect(store.get().aliasMerges).toEqual([]);
    expect(store.get().genderOverrides).toEqual([]);
    expect(store.get().voiceOverrides).toEqual([]);
    expect(store.get().dirty).toBe(false);
  });

  test("addMerge sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    expect(store.get().aliasMerges).toEqual([{ from: "Lizzy", to: "Elizabeth" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("setGender sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.setGender("elizabeth", "female");
    expect(store.get().genderOverrides).toEqual([{ characterId: "elizabeth", gender: "female" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("setVoice sets dirty flag", () => {
    const store = createCorrectionsStore();
    store.setVoice("elizabeth", "female_adult_01");
    expect(store.get().voiceOverrides).toEqual([{ characterId: "elizabeth", voiceId: "female_adult_01" }]);
    expect(store.get().dirty).toBe(true);
  });

  test("markSaved clears dirty flag and records saved corrections", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    store.markSaved(["ch01", "ch02"]);
    expect(store.get().dirty).toBe(false);
    expect(store.get().affectedChapters).toEqual(["ch01", "ch02"]);
    expect(store.get().savedCorrections).toEqual({
      aliasMerges: [{ from: "Lizzy", to: "Elizabeth" }],
      genderOverrides: [],
      voiceOverrides: [],
    });
  });

  test("setGender replaces existing override for same character", () => {
    const store = createCorrectionsStore();
    store.setGender("elizabeth", "female");
    store.setGender("elizabeth", "male");
    expect(store.get().genderOverrides).toEqual([{ characterId: "elizabeth", gender: "male" }]);
  });

  test("addMerge deduplicates by from", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    store.addMerge({ from: "Lizzy", to: "Beth" });
    expect(store.get().aliasMerges).toEqual([{ from: "Lizzy", to: "Beth" }]);
  });

  test("setVoice replaces existing override for same character", () => {
    const store = createCorrectionsStore();
    store.setVoice("elizabeth", "female_adult_01");
    store.setVoice("elizabeth", "male_adult_01");
    expect(store.get().voiceOverrides).toEqual([{ characterId: "elizabeth", voiceId: "male_adult_01" }]);
  });

  test("reset clears all state", () => {
    const store = createCorrectionsStore();
    store.addMerge({ from: "Lizzy", to: "Elizabeth" });
    store.setGender("elizabeth", "female");
    store.markSaved(["ch01"]);
    store.reset();
    expect(store.get()).toEqual(createCorrectionsStore().get());
  });

  test("subscribe returns unsubscribe and calls listener on change", () => {
    const store = createCorrectionsStore();
    let calls = 0;
    const unsub = store.subscribe(() => { calls++; });
    store.addMerge({ from: "A", to: "B" });
    expect(calls).toBe(1);
    unsub();
    store.addMerge({ from: "C", to: "D" });
    expect(calls).toBe(1);
  });
});
