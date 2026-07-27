import type { CharacterMeta, VoiceMeta, VoiceOption } from "../types";

const AUTO_CHARACTER_VOICE_PREFIX = "character_auto_";

const VOICE_NAMES_ZH: Record<string, string> = {
  narrator_default: "旁白（温暖女声）",
  female_adult_01: "女声 — 温暖有表现力",
  female_adult_02: "女声 — 明亮清澈",
  female_adult_03: "女声 — 温柔细腻",
  female_adult_04: "女声 — 活泼有活力",
  female_adult_05: "女声 — 从容优雅",
  male_adult_01: "男声 — 低沉浑厚",
  male_adult_02: "男声 — 清晰利落",
  male_adult_03: "男声 — 温暖亲切",
  male_adult_04: "男声 — 坚定威严",
  male_adult_05: "男声 — 平静醇厚",
  female_british_01: "女声 — 英式明亮",
  female_british_02: "女声 — 英式优雅",
  male_british_01: "男声 — 英式精致",
  male_british_02: "男声 — 英式温暖",
  neutral_dialogue_01: "中性对白",
};

export function localizeVoiceDisplayName(displayName: string, voiceId?: string): string {
  if (voiceId && VOICE_NAMES_ZH[voiceId]) return VOICE_NAMES_ZH[voiceId];
  return displayName;
}

interface BuildVoiceOptionsInput {
  providerVoices: VoiceMeta[];
  analysisVoices: VoiceMeta[];
  characters: CharacterMeta[];
}

export function buildVoiceOptions({
  providerVoices,
  analysisVoices,
  characters,
}: BuildVoiceOptionsInput): VoiceOption[] {
  const options = new Map<string, VoiceOption>();
  const catalog = providerVoices.length > 0 ? providerVoices : analysisVoices;

  for (const voice of catalog) {
    if (!voice.id || options.has(voice.id)) continue;
    options.set(voice.id, {
      id: voice.id,
      displayName: localizeVoiceDisplayName(voice.displayName || voice.id, voice.id),
      available: true,
    });
  }

  // Keep persisted assignments visible when a legacy script references a
  // voice that the active provider no longer exposes.
  for (const voice of analysisVoices) {
    if (!voice.id || options.has(voice.id)) continue;
    options.set(voice.id, {
      id: voice.id,
      displayName: localizeVoiceDisplayName(voice.displayName || voice.id, voice.id),
      available: false,
    });
  }

  for (const character of characters) {
    if (!character.voiceId) continue;
    if (
      character.voiceSource === "auto" &&
      character.voiceId.startsWith(AUTO_CHARACTER_VOICE_PREFIX)
    ) {
      options.set(character.voiceId, {
        id: character.voiceId,
        displayName: "角色自动音色（身份生成）",
        available: true,
      });
      continue;
    }
    if (options.has(character.voiceId)) continue;
    options.set(character.voiceId, {
      id: character.voiceId,
      displayName: "当前音色（不可用）",
      available: false,
    });
  }

  return [...options.values()];
}
