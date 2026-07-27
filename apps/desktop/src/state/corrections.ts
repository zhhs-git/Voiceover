import { create } from "zustand";

export interface AliasMerge {
  from: string;
  to: string;
}

export interface GenderOverride {
  characterId: string;
  gender: string;
}

export interface VoiceOverride {
  characterId: string;
  voiceId: string;
}

export interface CorrectionSet {
  aliasMerges: AliasMerge[];
  genderOverrides: GenderOverride[];
  voiceOverrides: VoiceOverride[];
}

export interface CorrectionState extends CorrectionSet {
  dirty: boolean;
  savedCorrections: CorrectionSet | null;
  affectedChapters: string[];
}

interface CorrectionActions {
  addMerge: (merge: AliasMerge) => void;
  setGender: (characterId: string, gender: string) => void;
  setVoice: (characterId: string, voiceId: string) => void;
  markSaved: (affectedChapters: string[]) => void;
  reset: () => void;
}

type CorrectionStore = CorrectionState & CorrectionActions;

function makeStore() {
  return create<CorrectionStore>()((set) => ({
    aliasMerges: [],
    genderOverrides: [],
    voiceOverrides: [],
    dirty: false,
    savedCorrections: null,
    affectedChapters: [],

    addMerge: (merge) =>
      set((s) => ({
        aliasMerges: [
          ...s.aliasMerges.filter((m) => m.from !== merge.from),
          merge,
        ],
        dirty: true,
      })),

    setGender: (characterId, gender) =>
      set((s) => ({
        genderOverrides: [
          ...s.genderOverrides.filter(
            (o) => o.characterId !== characterId,
          ),
          { characterId, gender },
        ],
        dirty: true,
      })),

    setVoice: (characterId, voiceId) =>
      set((s) => ({
        voiceOverrides: [
          ...s.voiceOverrides.filter(
            (o) => o.characterId !== characterId,
          ),
          { characterId, voiceId },
        ],
        dirty: true,
      })),

    markSaved: (affectedChapters) =>
      set((s) => ({
        dirty: false,
        savedCorrections: {
          aliasMerges: s.aliasMerges,
          genderOverrides: s.genderOverrides,
          voiceOverrides: s.voiceOverrides,
        },
        affectedChapters,
      })),

    reset: () =>
      set({
        aliasMerges: [],
        genderOverrides: [],
        voiceOverrides: [],
        dirty: false,
        savedCorrections: null,
        affectedChapters: [],
      }),
  }));
}

export function createCorrectionsStore() {
  const useStore = makeStore();

  return {
    get(): CorrectionState {
      const full = useStore.getState() as CorrectionStore;
      const {
        addMerge: _am,
        setGender: _sg,
        setVoice: _sv,
        markSaved: _ms,
        reset: _r,
        ...state
      } = full;
      return state as CorrectionState;
    },

    subscribe(fn: () => void) {
      return useStore.subscribe(fn);
    },

    addMerge: (merge: AliasMerge) =>
      useStore.getState().addMerge(merge),

    setGender: (characterId: string, gender: string) =>
      useStore.getState().setGender(characterId, gender),

    setVoice: (characterId: string, voiceId: string) =>
      useStore.getState().setVoice(characterId, voiceId),

    markSaved: (affectedChapters: string[]) =>
      useStore.getState().markSaved(affectedChapters),

    reset: () => useStore.getState().reset(),
  };
}

export const useCorrectionStore = makeStore();
