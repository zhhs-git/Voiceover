from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable
from urllib import request as urllib_request
from urllib.error import HTTPError

from audiobook_worker.audio_asset_ids import normalize_audio_plan_asset_ids
from audiobook_worker.dialogue import resolve_text_language, segment_dialogue
from audiobook_worker.llm_env import read_llm_environment


@dataclass(frozen=True)
class CharacterContext:
    """A character already identified in a previous chapter, passed for consistency."""
    id: str
    canonical_name: str
    aliases: list[str]
    gender: str
    age_class: str = "unknown"
    voice_design: str = ""


@dataclass(frozen=True)
class ChapterAnalysisRequest:
    book_id: str
    chapter_id: str
    text: str
    language: str
    known_characters: list[CharacterContext] = field(default_factory=list)
    stage_callback: Callable[[str, dict[str, Any]], None] | None = None
    # Completed stage payloads loaded from the chapter analysis cache.  They
    # are only reused when ``resume_from_stage`` explicitly says where to
    # continue; a normal analysis always starts a fresh three-stage run.
    cached_stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    resume_from_stage: str | None = None


@dataclass(frozen=True)
class AudioPlanningRequest:
    """Inputs for the split post-TTS scene, music, and SFX planning stages."""

    book_id: str
    chapter_id: str
    text: str
    language: str
    segments: list[dict[str, Any]]
    transcript: list[dict[str, Any]]
    # This is the active chapter cast, not the full book character roster.
    # Callers should use ``select_active_audio_characters`` when constructing
    # the request.  The stage payload applies the same guard as a second line
    # of defence for older/custom callers.
    characters: list[dict[str, Any]] = field(default_factory=list)
    # Audio planning is split into smaller, resumable requests. These fields
    # remain optional so custom analyzers and older callers keep the old API.
    stage_callback: Callable[[str, dict[str, Any]], None] | None = None
    cached_stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    resume_from_stage: str | None = None


_AUDIO_PLANNER_NON_CHARACTER_SPEAKERS = frozenset({"", "narrator", "unknown"})


def select_active_audio_characters(
    characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a compact roster containing only characters speaking this chapter.

    Chapter scripts intentionally retain the whole-book roster so character
    identity and fixed voices remain stable across chapters.  Audio planning
    does not need inactive characters, so only IDs referenced by the current
    segment speaker IDs are allowed through.  We also whitelist the fields
    that the planner can use; this prevents unrelated script metadata from
    inflating the request again.
    """

    if not isinstance(characters, list) or not isinstance(segments, list):
        return []

    active_ids = {
        str(segment.get("speakerId") or "").strip()
        for segment in segments
        if isinstance(segment, dict)
    }
    active_ids.difference_update(_AUDIO_PLANNER_NON_CHARACTER_SPEAKERS)
    if not active_ids:
        return []

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for character in characters:
        if not isinstance(character, dict):
            continue
        character_id = str(character.get("id") or "").strip()
        if not character_id or character_id not in active_ids or character_id in seen_ids:
            continue

        compact: dict[str, Any] = {"id": character_id}
        for key in ("canonicalName", "gender", "ageClass", "voiceId"):
            value = character.get(key)
            if value is not None and str(value).strip():
                compact[key] = str(value)

        aliases = character.get("aliases")
        if isinstance(aliases, list):
            compact["aliases"] = [str(alias) for alias in aliases if str(alias).strip()]

        voice_design = character.get("voiceDesign") or character.get("voiceDescription")
        if voice_design is not None and str(voice_design).strip():
            compact["voiceDesign"] = str(voice_design).strip()

        selected.append(compact)
        seen_ids.add(character_id)

    return selected


@dataclass(frozen=True)
class CharacterAnalysis:
    id: str
    canonical_name: str
    aliases: list[str]
    gender: str
    age_class: str
    confidence: float
    voice_design: str = ""


@dataclass(frozen=True)
class SegmentAnnotation:
    segment_index: int
    speaker_id: str
    emotion: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    pace: str = "normal"


@dataclass(frozen=True)
class MusicPlan:
    model: str
    duration_seconds: float
    prompt: str
    negative_prompt: str
    reason_zh: str


@dataclass(frozen=True)
class MusicVariantPlan:
    """One same-theme Stable Audio music variant for a scene."""

    id: str
    level: str
    model: str
    duration_seconds: float
    prompt: str
    negative_prompt: str
    reason_zh: str


@dataclass(frozen=True)
class MusicCuePlan:
    """A source-segment range that selects one music variant."""

    id: str
    start_segment_index: int
    end_segment_index: int
    variant_id: str
    reason_zh: str


@dataclass(frozen=True)
class MusicBreakPlan:
    """A short, intentional music-only breathing window."""

    after_segment_index: int
    duration_seconds: float
    reason_zh: str


@dataclass(frozen=True)
class SfxPlan:
    id: str
    model: str
    anchor_segment_index: int
    timing: str
    event_zh: str
    duration_seconds: float
    prompt: str
    negative_prompt: str
    reason_zh: str
    anchor_text: str = ""


@dataclass(frozen=True)
class AudioScenePlan:
    id: str
    start_segment_index: int
    end_segment_index: int
    summary_zh: str
    energy_arc: str = ""
    music: MusicPlan | None = None
    music_palette: dict[str, str] = field(default_factory=dict)
    music_variants: list[MusicVariantPlan] = field(default_factory=list)
    music_cues: list[MusicCuePlan] = field(default_factory=list)
    music_breaks: list[MusicBreakPlan] = field(default_factory=list)
    sfx: list[SfxPlan] = field(default_factory=list)


@dataclass(frozen=True)
class ChapterAudioPlan:
    scenes: list[AudioScenePlan] = field(default_factory=list)
    version: int = 1


@dataclass(frozen=True)
class ChapterAnalysisResult:
    characters: list[CharacterAnalysis]
    segment_annotations: list[SegmentAnnotation]
    audio_plan: ChapterAudioPlan = field(default_factory=ChapterAudioPlan)
    voice_directions: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3
    # OpenAI-compatible gateways do not all implement response_format in the
    # same way. Some Codex-backed routes accept ordinary chat requests but
    # fail or misreport JSON-mode requests. Keep JSON enforcement in the
    # prompt and local decoder by default; providers can opt in when this
    # capability is known to work.
    supports_response_format: bool = False


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]

_ENGLISH_SYSTEM_PROMPT = """\
You are an audiobook script analyst. Your job is to analyse a book chapter and produce a \
structured JSON identifying speaking characters and annotating each dialogue segment.

This is an identity-resolution task as well as a dialogue-tagging task. Read the entire
chapter before assigning any speaker. The `knownCharacters` list is the book-level roster;
its IDs are opaque registry keys, not names to reinterpret.

## Mandatory reasoning workflow
Do this silently before producing JSON:
1. Read all of `chapterText`, all `segments`, and all `knownCharacters`.
2. Mark scene boundaries, explicit speech tags, actions around quotations, pronouns,
   forms of address, and question/answer relationships.
3. Compare every possible speaker with the complete known roster before considering a
   new character.
4. Build a local speaker table and assign segments using semantic continuity, not line
   number or alternating turns.
5. Perform a final consistency pass: every non-narrator `speakerId` must be an exact ID
   in `characters` or `knownCharacters`, and every active character must be referenced
   by at least one segment.

## Cross-chapter identity rules (highest priority)
- If a person matches `knownCharacters`, reuse that character's exact `id` byte-for-byte
  and exact `canonicalName`. Never invent, translate, normalize, or replace a known ID.
- Match in this order: exact known ID; exact canonical name; explicit full-name,
  surname, nickname, or alias; then a title or kinship term only when scene context
  leaves exactly one compatible known candidate.
- A generic title or kinship term such as "Miss", "madam", "mother", or "father" is
  not a unique person. Do not merge or create a character from a generic term alone.
  If several known candidates fit, lower confidence and use `speaker_ambiguous` or
  `unknown` rather than inventing a duplicate.
- A different spelling, shortened name, title, or temporary model label is not evidence
  of a new person. Create a new character only when the chapter positively establishes
  that the person is not in the known roster.
- Add an alias only when the text explicitly proves it refers to the same person. Do
  not add every title, pronoun, or speculative relationship as an alias.
- If two people share a name, keep them separate when scene, relationship, age,
  location, or explicit context proves they are different; never merge by name alone.
- Do not copy inactive known characters into `characters`; an active known character
  must be included there and referenced with its original ID.

## Rules for characters
- `id` is only a temporary candidate key in this response. The worker assigns the \
  book-scoped registry ID after matching the candidate to the saved character roster.
- For a new English character, use a readable snake_case candidate key derived from \
  the character's most common name (e.g. "elizabeth_bennet", "mr_darcy").
- For a character in `knownCharacters`, reuse its exact `id` so the worker can \
  resolve the segment to the saved registry identity. NEVER invent a replacement \
  id for a known character.
- `canonicalName` is the full display name used in the book (e.g. "Elizabeth Bennet").
- List all known shorter forms, nicknames, and titles in `aliases` \
  (e.g. ["Lizzy", "Miss Bennet", "Eliza"]).
- `gender`: "female" | "male" | "neutral" | "unknown". Infer from pronouns and context.
- `ageClass`: "child" | "young" | "adult" | "older" | "unknown". Treat family
  identity as evidence: grandparents/elderly people are usually `older`, parents
  are usually `adult`, and sons/daughters/teenagers are usually `young` unless
  the chapter gives contrary age evidence. Preserve a known character's age
  class when the new chapter gives no explicit reason to change it.
- `confidence`: 0.0–1.0, reflect how sure you are the character is correctly identified.
- If a character already appears in `knownCharacters`, reuse their exact `id` and \
  `canonicalName`. Do NOT create a new entry for the same person.

## Speaker attribution rules
- Resolve explicit speech tags and nearby actions before using turn-taking. The tag and
  action outside quotation marks remain narration.
- Resolve pronouns such as "he" and "she" to an established person; never create a
  character whose canonical name is a pronoun.
- Quoted names, titles, labels, book titles, letters, announcements, poems, citations,
  remembered words, and reported speech are not automatically direct dialogue. A
  `type="narration"` segment with `quoted_material` must remain `narrator`.
- Inner monologue belongs to a clearly established viewpoint character. If the viewpoint
  is uncertain, use `narrator` or `unknown` with lower confidence.
- Never assign speakers by mechanical A-B-A-B alternation. One speaker may have several
  consecutive lines. Switch only for an explicit tag, direct answer/rebuttal/question,
  change of address, or clear semantic change.
- After a scene, time jump, flashback, or recalled event changes, rebuild the local
  speaker table while retaining the book-level roster.
- For anonymous groups, use the smallest stable set of scene roles and reuse them; do
  not create one new person per line. Use `unknown` only when no stable scene identity
  can be established.
- `speakerHint` and the pre-segmented `type` are weak evidence. If they conflict with
  the chapter, follow the chapter and add `speaker_hint_conflict`.
- When attribution is genuinely ambiguous, lower confidence and add one of
  `speaker_ambiguous`, `speaker_unknown`, `pronoun_ambiguous`, or
  `turn_taking_inferred` instead of guessing with high confidence.

## Rules for segmentAnnotations
- Annotate every segment in the input, including narration (speakerId = "narrator").
- For dialogue with a known speaker, use their `id` from the characters list.
- For dialogue with no identifiable speaker, use speakerId = "unknown".
- A segment marked `type="narration"` with warning `quoted_material` is quoted
  material such as a name, title, label, citation, remembered phrase, or sound
  effect. It must use speakerId = "narrator" and must never create or use a
  character voice for that text.
- `emotion`: "neutral" | "happy" | "sad" | "angry" | "afraid" | "tense" | "teasing" | \
  "whispering" | "excited" | "tired" | "grief" | "cold" | "pleading" | "surprised" | \
  "gentle" | "resolute" | "nervous" | "contemptuous" | "solemn" | "bitter". \
  Choose the most contextually appropriate.
- Emotion selection guidance:
  - `teasing`: mockery, sarcasm, taunting, belittling, or playful contempt. Do not collapse into `happy`.
  - `grief`: raw, overwhelming sorrow beyond `sad` — sobbing, wailing, bereavement.
  - `cold`: emotionally flat and detached, deliberate emotional absence, not just calm.
  - `pleading`: desperate entreaty or begging, vulnerability fully exposed.
  - `surprised`: sudden shock or disbelief, not joyful — use `excited` for positive surprise.
  - `gentle`: tender, soft, soothing address, especially to someone vulnerable.
  - `resolute`: firm, unwavering declaration or determination, heavier than `tense`.
  - `nervous`: inner anxiety and unease, distinct from external `tense` pressure.
  - `contemptuous`: cold, humourless disdain from above, heavier and colder than `teasing`.
  - `solemn`: grave, ceremonious, befitting oaths, eulogies, or momentous declarations.
  - `bitter`: suppressed resentment and grievance, not open `angry`.
- `pace`: "slow" | "normal" | "fast". Apply the mandatory pace rubric below to both \
  narration and dialogue; narration may be slow, normal, or fast when the text supports \
  a change, and must not be forced to "normal". Judge every segment independently; do not \
  mechanically use the same pace for every speaker or for the whole chapter.
- For narration, use the rhythm and delivery cues in the prose: descriptive or reflective \
  passages are normally `normal`, deliberate emphasis, grief, hesitation, or solemn reading \
  may be `slow`, and urgent action, rapid developments, or compressed narration may be `fast`.
- Pace means delivery speed, not volume, emotion, age, gender, social status, or \
  the inherent sound of a voice. Do not infer a slow pace merely because a topic is \
  serious, a sentence is long, or a speaker sounds authoritative.
- Use `normal` as the default when there is no clear speed cue. Everyday conversation, \
  explanations, questions, comments, greetings, gossip, and ordinary group chatter \
  are `normal`.
- Use `fast` only when the text or nearby action clearly shows urgency or compressed \
  delivery: rapid question-and-answer, interruption, rushing, panic, pursuit, excited \
  speech, shouting, or someone speaking before another person finishes.
- Use `slow` only when the text or nearby action clearly shows deliberate restraint: \
  long pauses, hesitation, grief, exhaustion, fear, solemn reading, careful emphasis, \
  or a warning spoken cautiously. A whisper is about volume; it is not automatically slow.
- For anonymous tea-house, street, or crowd dialogue, ordinary banter is `normal`; \
  quick back-and-forth is `fast`; only an explicitly cautious warning or hesitant line \
  is `slow`. Never make the other lines in a scene slow just because one line is a warning.
- When cues conflict, prioritize explicit speech/action descriptions, then punctuation \
  and sentence rhythm, then the broader subject matter. If evidence remains ambiguous, \
  choose `normal`.
- `confidence`: how sure you are about the speaker attribution (0.0–1.0).
- Only add `warnings` for genuine ambiguity (e.g. ["speaker_ambiguous"]).

## Final self-check
Before returning JSON, check: did an existing character get duplicated; did a title,
pronoun, quoted word, place, or object become a character; did any dialogue get assigned
by mechanical alternation; and does every non-narrator ID exist in the roster?

## Output format
Return a single JSON object with exactly two keys: `characters` and `segmentAnnotations`. \
Each character contains `id`, `canonicalName`, `aliases`, `gender`, `ageClass`, and `confidence`. \
Each segment annotation contains `segmentIndex`, `speakerId`, `emotion`, `pace`, `confidence`, \
and `warnings`. \
No markdown, no extra commentary — only the JSON object.
"""

_CHINESE_SYSTEM_PROMPT = """\
你是中文有声书剧本分析器。你的任务是先通读完整章节，再识别真正需要独立配音的角色，并为输入中的每个文本片段判断说话人和情绪。输入中的预切分结果与 speakerHint 只是候选提示，不是事实；必须以章节原文、上下文和人物关系为准。

这个任务同时是“说话人归属”和“全书角色身份消歧”。请只在内部执行以下流程，不要输出推理：
1. 阅读完整的 `chapterText`、全部 `segments` 和全部 `knownCharacters`。
2. 先标出场景边界、明确发言标签、动作描写、代词指向、称呼、问答关系和引号范围。
3. 对每个可能说话人，先与完整的 `knownCharacters` 名册逐项比对；确认不匹配后才允许新建角色。
4. 在当前场景建立“片段 -> 说话人”表，再逐条输出标注；不能根据片段序号机械轮换。
5. 输出前检查：每个非旁白 `speakerId` 必须是 `characters` 或 `knownCharacters` 中的精确 ID；本章实际发言的角色必须至少出现在一个片段中；不能出现同一人物的重复记录。

## 跨章节角色身份规则（最高优先级）
1. `knownCharacters` 是本书角色名册，里面的 ID 是不透明的系统注册 ID。已有角色必须逐字复用原 ID，不能改大小写、改拼音、翻译、截断或重新生成 ID。
2. 按以下顺序匹配已有角色：精确匹配已知 ID；精确匹配 `canonicalName`；匹配正文明确证明等价的全名、姓氏、昵称、乳名或别名；最后才可在场景、性别、年龄、关系和动作共同表明“唯一候选”时使用亲属称谓或职称匹配。
3. “小姐、夫人、殿下、母亲、父亲、老师、掌柜”等泛称本身不是唯一人物。不能仅凭泛称合并角色，也不能仅凭泛称新建角色；存在多个候选时宁可降低置信度并使用 `speaker_ambiguous` 或 `unknown`，不要制造新的重复角色。
4. 只要一个人物可能是名册中的已有角色，就必须优先复用已有角色。不同写法、简称、尊称或模型临时 ID，不构成新人物证据。只有正文提供了“名册中不存在这个人”的正面证据时，才新建角色。
5. 新角色的 `id` 只是本次响应的候选键，最终 ID 由 Worker 生成；不能把模型候选 ID 当成全书永久 ID。已有角色必须保留名册中的 `canonicalName`，不能因为本章出现简称或尊称就改名。
6. `aliases` 只能收录正文明确证明属于同一人物的简称、昵称、姓氏、尊称或亲属称谓；不能把所有出现过的称呼、代词或猜测关系都加入别名。
7. 如果正文明确存在同名不同人，必须结合场景、地点、关系、年龄、性别或动作区分，不能只因姓名相同就合并。
8. `knownCharacters` 中本章没有发言的角色不要复制到 `characters`；本章发言的已知角色必须出现在 `characters` 中，并在片段中使用原始 ID。

## 总体原则
1. 先阅读 chapterText 的完整上下文，再逐个标注 segments；不得只看单句猜角色。
2. 必须为每个 segmentIndex 返回且只返回一条 segmentAnnotations，顺序和索引不得改变。
3. 旁白使用 speakerId="narrator"。对于需要合成有声书的直接对白，必须尽量提供可分配声音的角色；只有既无法判断真实人物、也无法建立匿名场景身份的单句零散对白才使用 speakerId="unknown"。
4. 不得把地点、组织、书名、物品、章节标题或仅被提及但没有发言的人登记为配音角色。
5. speakerHint 仅作为证据之一。如果它与完整上下文冲突，以完整上下文为准，并添加 warning="speaker_hint_conflict"。
6. 预切分的 type 也只是候选：普通旁白候选可以被判为角色的内心独白或无引号直接引语；但标记了 quoted_material 的片段必须保持 narrator，不能被改成角色对白。

## 中文说话人归属规则
1. 优先使用明确发言标签，例如“张三说/问/答/喊”“母亲叹道”“李婶低声说”，标签可能位于引号前或引号后。
2. 发言标签省略姓名时，结合紧邻动作、代词指向、当前场景人物和上一轮发言判断。“他/她”必须先解析到已出现人物，不能把“他”或“她”创建成新角色。
3. 连续对白应先根据问答关系、称呼、语气承接和共同话题判断参与人数与说话人。只有在上下文明确只有两名参与者、轮换没有被动作或新人物打断时，才可把轮流发言作为辅助证据；不得仅凭对白轮流出现机械分配说话人，也不得忽略同一人物连续说两句的可能。
   判断是否换人时，语义关系优先于行号交替：后一句若是在补充前一句的观察、列举同一现象、延续同一语气或完成同一个表达，应优先归给同一人物；只有回答问题、反驳、追问、回应称呼或明显改变立场时，才有较强的换人证据。

   【关键反例】茶肆中连续出现以下对白：
   "白幡是做什么？"（问）
   "嗬，官老爷都系白腰带？"（补充同一观察）
   "你几日没出门了，长公主薨了啊！"（解释回答）
   "她死了不是好事吗？该敲锣打鼓庆贺才是啊。"（评论反应）
   "嘘……这话被官差听见，可要抓你坐牢的。"（警告）

   ❌ 错误：机械交替分配为"甲乙甲乙甲"
   ✅ 正确：语义优先分配为"甲甲乙甲乙"（第1、2句都是甲在观察白色丧仪，第3句乙解释，第4句甲评论，第5句乙警告）
4. 同一人物的姓名、姓氏、昵称、乳名、尊称、职位和亲属称谓应合并，例如“张建国/老张/张叔”“王秀兰/母亲/王婶”。有歧义的泛称不得强行合并。
5. 引号中的称谓、名字、标签、书名、信件、公告、诗句、转述或回忆中的引用，不等于现场角色直接发言；若由叙述者朗读，使用 narrator。像“自己打生下来就被称‘殿下’，何时被人称过‘小姐’?”这样的句子中，两个引号内只是被提及的称谓，必须和整句一样归旁白。
6. 内心独白归属于明确的视角人物；无法确认视角人物时使用 narrator 或 unknown，并降低 confidence。
7. 引号外的发言标签及动作描写属于旁白，不能并入角色朗读文本。
8. 场景切换、时间跳转、回忆开始或结束后，重新判断在场人物，不延续上一场景的轮流发言假设。
9. 对茶肆闲谈、街头议论、士兵交谈等匿名连续对白，应创建可独立配音的场景角色，例如“茶客甲”“茶客乙”“士兵甲”，并在同一场景中稳定复用。角色数量必须采用能解释完整场景的最小合理人数；上下文表明是两人对话时，只能复用“甲/乙”两名角色，不得为每句对白各创建一个新角色，也不得擅自增加“丙/丁/戊”。连续观察或补充说明可以由同一人连说两句；问答往返再结合语义切换角色。只有无法建立任何稳定场景身份的孤立对白才使用 unknown。
10. 输入可能来自 OCR，存在左右引号颠倒、弯引号与直引号混用、缺失闭引号、错别字和异常换行。应依据语义、段落和发言标签恢复对白边界，不得把数段对白或后续旁白合并给同一个角色。
11. `speakerHint` 与完整上下文冲突时，以完整上下文为准，并添加 `speaker_hint_conflict`；不能为了迎合预切分提示而改变已知角色身份。
12. 如果两个已知角色都可能是说话人，不要擅自新建第三个角色；选择证据更强的候选并降低 confidence，或使用 `unknown`/`speaker_ambiguous`。只有正文证据表明是新人物时才新增角色。

## 角色规则
1. 中文人物首次出现时，`id` 只是本次响应内的候选键，通常使用最稳定的 canonicalName 中文姓名本身，例如“张三”；不要自行生成不稳定的拼音 ID。最终角色 ID 由 Worker 的全书角色注册表生成。
2. 如果人物已在 knownCharacters 中，必须复用其完全相同的 id 和 canonicalName；别名命中时不得创建新人物。
3. canonicalName 使用作品中最明确、最稳定的姓名；aliases 收录实际出现的简称、昵称、尊称和称谓。
4. gender 只能是 female、male、neutral、unknown。根据明确称谓、代词和上下文判断；证据不足时必须用 unknown，不要凭姓名刻板猜测。
5. ageClass 只能是 child、young、adult、older、unknown；证据不足时使用 unknown。亲属身份是重要证据：祖父母、老人、老者通常为 older；父亲、母亲、叔伯、师父等通常为 adult；儿子、女儿、少爷、小姐、少年、少女在没有相反年龄证据时通常为 young；明确儿童、幼童、孩子为 child。若已有 knownCharacters 的年龄信息且上下文没有新的明确反证，保持其 ageClass，不能因单句语气或称谓随意改动。
6. characters 仅包含本章实际发言或发生内心独白、需要独立配音的人物；包括有稳定场景身份的匿名发言者，但不包括没有发言的背景群众。

## 情绪、语速、置信度与警告
emotion 只能是以下20种：neutral、happy、sad、angry、afraid、tense、teasing、whispering、excited、tired、grief、cold、pleading、surprised、gentle、resolute、nervous、contemptuous、solemn、bitter。根据完整上下文、当前句与相邻动作判断，证据不足时用 neutral。
情绪选择指引：
- teasing：戏谑、嘲弄、挖苦、轻蔑挑衅或带笑的贬低，不能因语气轻快误判为 happy，如"哟，还挺乖""废物，喂狗么"应优先用 teasing。
- grief：超出 sad 的原始悲恸，如嚎啕大哭、突闻噩耗、撕裂性的痛苦。
- cold：刻意的情感疏离与冰冷，不是平静，而是主动隔绝情感，语气毫无温度。
- pleading：极度恳切的哀求，带着脆弱与迫切，比求人还要低微一层。
- surprised：突然的震惊或难以置信，不是愉快的惊喜；愉快的惊喜用 excited 或 happy。
- gentle：轻柔温存地对待脆弱者或幼小者，带着体贴与安抚，比 happy 更克制更温和。
- resolute：坚定不移的表态或决意，比 tense 更有主动性，带着不可撼动的分量。
- nervous：内心的焦虑忐忑，与外部压力带来的 tense 不同，是一种向内的慌乱。
- contemptuous：冷蔑鄙视，不带笑意，比 teasing 更冷更刻薄，是居高临下的轻视。
- solemn：庄严肃穆，适合宣誓、讣告、重要宣告等需要郑重的场合。
- bitter：压抑的苦涩与积怨，不是公开的 angry，而是咽下去的辛酸与不甘。
pace 只能是 slow、normal、fast。旁白和对白都必须根据文本证据判断语速；旁白可以返回 slow、normal 或 fast，不能强制全部返回 normal。语速只表示说话的快慢节奏，不表示音量、音色、年龄、性别、身份地位或声音本身特质；不能因为内容严肃、句子较长、人物年长或身份威严就直接判为 slow。

### 语速判断标准（必须遵守，按优先级从高到低）

**第一优先：明确的文本/动作描写**
出现"急着说""抢着答""连声催""快步跑来说"等急促描写 → fast；出现"缓缓开口""慢慢说道""停顿了很久""哽咽着说""一字一顿"等放慢描写 → slow；短促连续的感叹与追问句群 → 倾向 fast；长句中多处省略号或逗号停顿 → 倾向 slow。

**第二优先：情绪对语速的默认倾向**
当没有明确文本/动作速度描写时，以情绪作为参考默认值，但可被句子节奏或上下文推翻。

倾向 slow 的情绪（无反例时选 slow）：
- grief：悲恸令语速破碎迟缓，除非是失控嚎哭的爆发句
- tired：身心俱疲导致说话拖沓费力
- solemn：郑重宣告或誓言需要每字有分量
- gentle：轻柔安慰不宜催促
- bitter：压抑的积怨往往低沉而缓，情绪崩溃时可转 normal
- cold：刻意疏离常拖长语速以示漠然，若是冰冷但干脆则用 normal
- pleading：恳切哀求多放慢以示郑重，急迫恳求可用 normal

倾向 fast 的情绪（无反例时选 fast）：
- excited：难掩激动时语速加快
- afraid（恐慌型）：慌乱逃跑、惊叫时倾向 fast；被吓呆或颤抖时倾向 slow，需结合上下文判断
- surprised（脱口型）：突然脱口反应倾向 fast；呆滞后缓慢确认倾向 slow
- nervous（慌乱型）：语无伦次的慌乱倾向 fast；犹豫停顿的忐忑倾向 slow

默认 normal 的情绪（无明确文本证据时不改变语速）：
- neutral、happy、tense、teasing、contemptuous、resolute、whispering
特别说明：angry 爆发型可 fast，冷怒或一字一顿可 slow，无法判断用 normal；tense 本身不影响语速；whispering 只说明音量，有犹豫停顿证据才用 slow；resolute 坚定有力不意味缓慢，默认 normal。

**第三优先：场景默认规则**
旁白没有明确速度证据时使用 normal；只有文本或动作明确支持时才使用 slow 或 fast。茶肆/街头/集市匿名闲谈：普通聊天用 normal，短促连续问答用 fast，只有明确低声警告或犹豫才用 slow，不能因一句警告将整场对白或旁白改为 slow。

每一句必须独立判断 pace，不能机械复制上一句的语速。冲突时：明确的动作描写 > 情绪默认倾向 > 场景规则；仍无法确定时选 normal。
confidence 为 0.0 到 1.0。明确姓名发言标签通常可高于 0.9；代词解析或双人轮换应降低；多名候选人时应低于 0.6。
只在真实歧义时添加 warnings，可用 speaker_ambiguous、speaker_unknown、pronoun_ambiguous、turn_taking_inferred、speaker_hint_conflict、quoted_material、inner_monologue_uncertain。

## 最终自检
输出前逐项检查：是否把已有角色误建成了新角色；是否把泛称、代词、被提及的名字、地点或物品当成了角色；是否把连续对白错误地机械交替；是否有同一人物的两个记录；每个 speakerId 是否都能在角色名册中找到。

## 示例
输入”张三说道：’走吧。’”时，”走吧。”属于张三，发言标签属于 narrator。
输入”’外面下雨呢。’李婶提醒道。”时，引号内容属于李婶，引号后的动作标签属于 narrator。
输入”真的要走吗？张三心里想。”时，内心独白属于张三；不能创建名为”他”的角色。
输入”桌上摊着《远方》：’明月照故乡。’”时，诗句属于引用材料，speakerId 使用 narrator，并添加 quoted_material。

## 输出格式
只返回一个 JSON 对象，不要 Markdown，不要解释。对象必须且只能包含 characters 和 segmentAnnotations 两个顶层字段。
characters 每项包含候选 id、canonicalName、aliases、gender、ageClass、confidence。候选 id 不是最终的全书角色 ID。
segmentAnnotations 每项包含 segmentIndex、speakerId、emotion、pace、confidence、warnings。
"""


def _system_prompt(language: str) -> str:
    return _CHINESE_SYSTEM_PROMPT if language.startswith("zh") else _ENGLISH_SYSTEM_PROMPT


class MockLLMAnalyzer:
    backend_id = "mock"
    supports_real_model = False

    def analyze_chapter(self, request: ChapterAnalysisRequest) -> ChapterAnalysisResult:
        request = replace(
            request,
            language=resolve_text_language(request.text, request.language),
        )
        segments = segment_dialogue(request.text, language=request.language)
        characters: dict[str, CharacterAnalysis] = {}

        # Seed with known characters so mock doesn't duplicate them
        for kc in request.known_characters:
            characters[kc.id] = CharacterAnalysis(
                id=kc.id,
                canonical_name=kc.canonical_name,
                aliases=kc.aliases,
                gender=kc.gender,
                age_class=kc.age_class,
                confidence=0.78,
            )

        annotations: list[SegmentAnnotation] = []

        for index, segment in enumerate(segments):
            if segment.type != "dialogue":
                continue

            speaker = segment.speaker_hint or "unknown"
            speaker_id = _speaker_id(speaker)

            # Check if known character matches (by name/alias)
            resolved_id = _resolve_known_character(speaker, request.known_characters)
            if resolved_id:
                speaker_id = resolved_id
            elif speaker != "unknown" and speaker_id not in characters:
                characters[speaker_id] = CharacterAnalysis(
                    id=speaker_id,
                    canonical_name=speaker,
                    aliases=[],
                    gender=_guess_gender(speaker),
                    age_class="adult",
                    confidence=0.78,
                )

            confidence = 0.76 if speaker != "unknown" else 0.35
            annotations.append(
                SegmentAnnotation(
                    segment_index=index,
                    speaker_id=speaker_id,
                    emotion=_guess_emotion(segment.text),
                    confidence=confidence,
                    warnings=[] if speaker != "unknown" else ["speaker_unknown"],
                )
            )

        return ChapterAnalysisResult(
            characters=list(characters.values()),
            segment_annotations=annotations,
        )

    def plan_audio(self, request: AudioPlanningRequest) -> ChapterAudioPlan:
        """Offline mode keeps audio planning empty instead of inventing assets."""
        return ChapterAudioPlan()


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    model_id: str
    base_url: str
    api_key: str
    api: str
    family: str
    max_tokens: int
    supports_response_format: bool = False


class OpenAICompatibleAnalyzer:
    supports_real_model = True

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or _post_json

    @property
    def backend_id(self) -> str:
        return self.config.provider

    def analyze_chapter(self, request: ChapterAnalysisRequest) -> ChapterAnalysisResult:
        from audiobook_worker.llm_stages import run_split_chapter_analysis

        try:
            return run_split_chapter_analysis(self, request)
        except RuntimeError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            # Keep the worker boundary stable for callers.  Stage parsers use
            # ValueError internally, while the old analyzer API exposed
            # analysis failures as RuntimeError.
            raise RuntimeError(f"Chapter analysis stage validation failed: {error}") from error

    def plan_audio(self, request: AudioPlanningRequest) -> ChapterAudioPlan:
        from audiobook_worker.llm_stages import run_audio_planning

        try:
            return run_audio_planning(self, request)
        except RuntimeError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Audio planning validation failed: {error}") from error

    def _request_stage_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        stage_name: str = "LLM stage",
    ) -> dict[str, Any]:
        """Send one narrowly scoped JSON request for an analysis stage."""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.1,
        }
        if self.config.supports_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        # DeepSeek's reasoning mode can spend the whole request budget on
        # internal reasoning for this large structured-analysis prompt. The
        # prompt already contains the required analysis rules, so disable
        # provider-specific reasoning for DeepSeek only.
        if self.config.provider.casefold() == "deepseek":
            payload["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._transport(
                    url,
                    {
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    payload,
                    self.config.timeout_seconds,
                )
                content = response["choices"][0]["message"]["content"]
                return _decode_model_json(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    # Ordinary errors use a short exponential backoff. A
                    # Codex-backed gateway can briefly lose its upstream auth
                    # lease and report that as 503/auth_unavailable; give it
                    # a longer recovery window before declaring the stage
                    # failed.
                    delay = (
                        5 * (2 ** attempt)
                        if _is_transient_auth_error(exc)
                        else 2 ** attempt
                    )
                    time.sleep(delay)
                    continue

        raise RuntimeError(
            f"{stage_name} failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error


def _is_transient_auth_error(error: Exception) -> bool:
    """Return whether a gateway error is likely recoverable by waiting."""

    if getattr(error, "code", None) == 503:
        return True
    detail = str(error).casefold()
    return "auth_unavailable" in detail or "http 503" in detail


def _decode_model_json(content: Any) -> dict[str, Any]:
    """Decode JSON returned by providers with minor wrapper variations.

    OpenAI-compatible providers normally return a JSON string, but some model
    versions wrap it in a Markdown fence or return an array of text parts.  A
    wrapper variation should not be confused with a chapter-analysis failure.
    """

    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        content = "".join(parts)
    if not isinstance(content, str):
        raise ValueError("LLM response content must contain a JSON object")

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Recover a single JSON object surrounded by a short accidental
        # preamble/postscript.  Do not attempt to repair arbitrary JSON.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("LLM response did not contain valid JSON") from error

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def default_analyzer(model_id: str | None = None):
    """Build the configured analyzer, optionally overriding the process default."""
    model_override = model_id or read_llm_environment().model_id
    if model_override == "mock":
        return MockLLMAnalyzer()
    resolved = resolve_model(model_override)
    analyzer = analyzer_from_resolved_model(resolved) if resolved else None
    if analyzer is not None:
        return analyzer
    return MockLLMAnalyzer()


def resolve_model(model_arg: str | None = None) -> ResolvedModel | None:
    environment = read_llm_environment()
    try:
        config = read_models_json()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # A project-local endpoint must stay usable even if an optional legacy
        # catalog was manually edited into invalid JSON.
        config = None
    if environment.base_url:
        selected_model = model_arg or environment.model_id
        if not selected_model and isinstance(config, dict):
            selected_model = str(config.get("default") or "").strip()
        if not selected_model:
            selected_model = "gpt-4o"
        metadata = _catalog_model_details(config, selected_model)
        provider = str(metadata.get("provider") or "env")
        provider_config = metadata.get("providerConfig")
        model_entry = metadata.get("model")
        if not isinstance(provider_config, dict):
            provider_config = {}
        if not isinstance(model_entry, dict):
            model_entry = {}
        return ResolvedModel(
            provider=provider,
            model_id=selected_model,
            base_url=environment.base_url,
            api_key=environment.api_key or "unused",
            api=str(provider_config.get("api") or "openai-completions"),
            family=str(provider_config.get("family") or "default"),
            max_tokens=int(model_entry.get("maxTokens", 8192)),
            supports_response_format=bool(
                model_entry.get(
                    "supportsResponseFormat",
                    provider_config.get("supportsResponseFormat", False),
                )
            ),
        )
    if config is not None:
        return resolve_model_from_config(config, model_arg)

    base_url = environment.base_url
    if not base_url:
        return None
    return ResolvedModel(
        provider="env",
        model_id=environment.model_id or "gpt-4o",
        base_url=base_url,
        api_key=environment.api_key or "unused",
        api="openai-completions",
        family="default",
        max_tokens=8192,
    )


def _catalog_model_details(
    config: dict[str, Any] | None,
    model_id: str,
) -> dict[str, Any]:
    """Find optional metadata without allowing catalog URLs/keys to win."""

    if not isinstance(config, dict):
        return {}
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return {}
    for provider, raw_provider in providers.items():
        if not isinstance(raw_provider, dict):
            continue
        models = raw_provider.get("models", [])
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            raw_id = str(model.get("id") or "").strip()
            candidates = {raw_id, f"{provider}/{raw_id}"}
            if model_id in candidates:
                return {
                    "provider": str(provider),
                    "providerConfig": raw_provider,
                    "model": model,
                }
    return {}


def read_models_json(paths: list[Path] | None = None) -> dict[str, Any] | None:
    search_paths = paths or [
        Path.home() / ".pi" / "agent" / "models.json",
        Path.home() / ".pi" / "models.json",
    ]
    for path in search_paths:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
    return None


def resolve_model_from_config(config: dict[str, Any], model_arg: str | None = None) -> ResolvedModel:
    lookup = model_arg or config.get("default")
    providers = config.get("providers", {})
    if lookup:
        provider_name, *segments = lookup.split("/")
        provider_config = providers.get(provider_name)
        if provider_config:
            model_entry, model_id = _find_model_entry(provider_name, provider_config, segments)
            return _resolved_model(provider_name, provider_config, model_entry, model_id)

        for candidate_provider, candidate_config in providers.items():
            for model in candidate_config.get("models", []):
                if model.get("id") == lookup:
                    return _resolved_model(candidate_provider, candidate_config, model, lookup)
            prefix = f"{candidate_provider}/"
            if lookup.startswith(prefix):
                stripped = lookup[len(prefix):]
                for model in candidate_config.get("models", []):
                    if model.get("id") == stripped:
                        return _resolved_model(candidate_provider, candidate_config, model, stripped)

    provider_name, provider_config = next(iter(providers.items()))
    model_entry = provider_config.get("models", [{}])[0]
    return _resolved_model(provider_name, provider_config, model_entry, model_entry.get("id", "default"))


def analyzer_from_models_config(
    config: dict[str, Any],
    model_arg: str | None = None,
    *,
    transport: Transport | None = None,
) -> OpenAICompatibleAnalyzer | MockLLMAnalyzer:
    analyzer = analyzer_from_resolved_model(
        resolve_model_from_config(config, model_arg),
        transport=transport,
    )
    return analyzer or MockLLMAnalyzer()


def analyzer_from_resolved_model(
    resolved: ResolvedModel,
    *,
    transport: Transport | None = None,
) -> OpenAICompatibleAnalyzer | None:
    if resolved.api != "openai-completions":
        return None
    return OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider=resolved.provider,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            model=resolved.model_id,
            max_tokens=resolved.max_tokens,
            supports_response_format=resolved.supports_response_format,
        ),
        transport=transport,
    )


def _find_model_entry(
    provider: str,
    provider_config: dict[str, Any],
    segments: list[str],
) -> tuple[dict[str, Any], str]:
    models = provider_config.get("models", [])
    resolved_model_id = "/".join(segments)
    full_model_id = f"{provider}/{resolved_model_id}" if resolved_model_id else provider
    for model in models:
        if model.get("id") == full_model_id:
            return model, full_model_id
    for start in range(0, len(segments) + 1):
        candidate = "/".join(segments[start:])
        for model in models:
            if model.get("id") == candidate:
                return model, candidate
    return {}, resolved_model_id


def _resolved_model(
    provider: str,
    provider_config: dict[str, Any],
    model_entry: dict[str, Any],
    model_id: str,
) -> ResolvedModel:
    return ResolvedModel(
        provider=provider,
        model_id=model_id,
        base_url=provider_config["baseUrl"],
        api_key=_resolve_api_key(provider_config),
        api=provider_config.get("api", "openai-completions"),
        family=provider_config.get("family", "default"),
        max_tokens=int(model_entry.get("maxTokens", 8192)),
        supports_response_format=bool(
            model_entry.get(
                "supportsResponseFormat",
                provider_config.get("supportsResponseFormat", False),
            )
        ),
    )


def _resolve_api_key(provider_config: dict[str, Any]) -> str:
    if provider_config.get("apiKey"):
        return str(provider_config["apiKey"])
    if provider_config.get("apiKeyEnv"):
        return os.environ.get(str(provider_config["apiKeyEnv"]), "")
    return "unused"


def _speaker_id(name: str) -> str:
    stripped = name.strip()
    if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff·]+", stripped):
        return stripped
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return normalized or "unknown"


def _resolve_known_character(name: str, known: list[CharacterContext]) -> str | None:
    """Return the id of a known character whose canonical name or alias matches."""
    name_lower = name.lower().strip()
    for kc in known:
        if kc.canonical_name.lower() == name_lower:
            return kc.id
        if any(alias.lower() == name_lower for alias in kc.aliases):
            return kc.id
    return None


def _analysis_prompt(request: ChapterAnalysisRequest) -> str:
    segments = segment_dialogue(request.text, language=request.language)
    segment_lines = [
        {
            "segmentIndex": index,
            "type": segment.type,
            "text": segment.text,
            "startOffset": segment.start_offset,
            "endOffset": segment.end_offset,
            "speakerHint": segment.speaker_hint,
            "warnings": segment.warnings,
        }
        for index, segment in enumerate(segments)
    ]

    payload: dict[str, Any] = {
        "chapterId": request.chapter_id,
        "language": request.language,
        "chapterText": request.text,
        "segments": segment_lines,
    }

    if request.known_characters:
        payload["knownCharacters"] = [
            {
                "id": kc.id,
                "canonicalName": kc.canonical_name,
                "aliases": kc.aliases,
                "gender": kc.gender,
                "ageClass": kc.age_class,
            }
            for kc in request.known_characters
        ]

    return json.dumps(payload, ensure_ascii=False)


_AUDIO_MODELS = {"sm-music", "sm-sfx"}
_AUDIO_TIMINGS = {"before", "during", "after"}
_MAX_AUDIO_DURATION_SECONDS = 120.0


def _audio_model(
    item: dict[str, Any],
    path: str,
    *,
    expected: str,
) -> str:
    """Normalize the planner's model label to a supported Stable Audio model.

    The planner has one supported model per asset kind.  Models sometimes
    return the friendly labels ``music``/``sfx`` or omit the field entirely;
    those should not silently erase an otherwise valid asset during validation.
    Manual Stable Audio scripts are still validated separately by
    ``stable_audio.py``.
    """

    value = item.get("model")
    if value is None:
        return expected
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"audio plan field {path}.model must be a string")
    # The planner cannot select another backend model: Stable Audio exposes
    # exactly one supported model for each asset kind. Treat the returned
    # label as a hint and bind the asset to that kind's supported model.
    return expected


def _audio_string(item: dict[str, Any], key: str, path: str, *, required: bool = True) -> str:
    value = item.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"audio plan field {path}.{key} must be a non-empty string")
    return value.strip()


def _audio_int(item: dict[str, Any], key: str, path: str) -> int:
    value = item.get(key)
    if isinstance(value, bool):
        raise ValueError(f"audio plan field {path}.{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            parsed = math.nan
        if math.isfinite(parsed) and parsed.is_integer():
            return int(parsed)
    raise ValueError(f"audio plan field {path}.{key} must be an integer")


def _audio_duration(item: dict[str, Any], path: str) -> float:
    value = item.get("durationSeconds")
    if isinstance(value, bool):
        raise ValueError(f"audio plan field {path}.durationSeconds must be a number")
    if isinstance(value, (int, float)):
        duration = float(value)
    elif isinstance(value, str):
        try:
            duration = float(value.strip())
        except ValueError as error:
            raise ValueError(
                f"audio plan field {path}.durationSeconds must be a number"
            ) from error
    else:
        raise ValueError(f"audio plan field {path}.durationSeconds must be a number")
    if not math.isfinite(duration) or not 0 < duration <= _MAX_AUDIO_DURATION_SECONDS:
        raise ValueError(
            f"audio plan field {path}.durationSeconds must be between 0 and "
            f"{_MAX_AUDIO_DURATION_SECONDS:g}"
        )
    return duration


def _optional_audio_string(
    item: dict[str, Any],
    key: str,
) -> str:
    """Read an optional text field without letting a model type error abort analysis."""

    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


_MUSIC_PROMPT_MAX_CHARACTERS = 260
_MUSIC_PROMPT_PREFIX = "TrackType: Music, VocalType: Instrumental"
_MUSIC_PROMPT_REDUNDANCY_PATTERNS = (
    r"\bno vocals\b",
    r"\bno lyrics\b",
    r"\bseamless loopable bed\b",
    r"\bseamless background bed\b",
    r"\bno abrupt ending\b",
    r"\bcontinuous audiobook background music bed\b",
    r"\bclearly audible but restrained under speech\b",
)


def _compact_music_prompt(value: Any) -> str:
    """Keep Stable Audio music prompts short and tag-led.

    Stable Audio's official prompt guide gets useful results from a compact
    combination of TrackType/VocalType, genre, instruments, mood/energy, and
    BPM.  The planner may still return verbose prose, so normalize redundant
    audiobook/mixing instructions before persisting or sending a prompt to
    Stable Audio.
    """

    text = " ".join(str(value or "").split()).strip()
    for pattern in _MUSIC_PROMPT_REDUNDANCY_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"(?:,\s*){2,}", ", ", text)
    text = text.strip(" ,.;，。；")

    has_track_type = re.search(r"\bTrackType\s*:\s*Music\b", text, re.IGNORECASE)
    has_vocal_type = re.search(
        r"\bVocalType\s*:\s*Instrumental\b", text, re.IGNORECASE
    )
    if not has_track_type:
        text = f"{_MUSIC_PROMPT_PREFIX}, {text}" if text else _MUSIC_PROMPT_PREFIX
    elif not has_vocal_type:
        text = re.sub(
            r"^\s*TrackType\s*:\s*Music\s*,?\s*",
            f"{_MUSIC_PROMPT_PREFIX}, ",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    if len(text) > _MUSIC_PROMPT_MAX_CHARACTERS:
        cutoff = text.rfind(",", 0, _MUSIC_PROMPT_MAX_CHARACTERS)
        if cutoff < len(_MUSIC_PROMPT_PREFIX):
            cutoff = _MUSIC_PROMPT_MAX_CHARACTERS
        text = text[:cutoff].rstrip(" ,.;，。；")
    return text


def _parse_music_plan(value: Any, path: str) -> MusicPlan:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    return MusicPlan(
        model=_audio_model(value, path, expected="sm-music"),
        duration_seconds=_audio_duration(value, path),
        prompt=_compact_music_prompt(_audio_string(value, "prompt", path)),
        negative_prompt=_optional_audio_string(value, "negativePrompt"),
        reason_zh=_optional_audio_string(value, "reasonZh"),
    )


def _parse_music_variant(value: Any, path: str) -> MusicVariantPlan:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    variant_id = _audio_string(value, "id", path)
    level = _optional_audio_string(value, "level").lower()
    if level not in {"low", "medium", "high"}:
        raise ValueError(f"{path}.level must be low, medium, or high")
    return MusicVariantPlan(
        id=variant_id,
        level=level,
        model=_audio_model(value, path, expected="sm-music"),
        duration_seconds=_audio_duration(value, path),
        prompt=_compact_music_prompt(_audio_string(value, "prompt", path)),
        negative_prompt=_optional_audio_string(value, "negativePrompt"),
        reason_zh=_optional_audio_string(value, "reasonZh"),
    )


def _parse_audio_plan(value: Any) -> ChapterAudioPlan:
    """Parse the optional audio plan defensively.

    Character and segment analysis is still useful when the model makes a type
    mistake in an optional Stable Audio field.  Invalid scenes/assets are
    discarded here and the core analysis continues; downstream validation still
    checks the surviving plan against the real segment timeline.
    """

    if value is None:
        return ChapterAudioPlan()
    if not isinstance(value, dict):
        return ChapterAudioPlan()

    raw_scenes = value.get("scenes", [])
    if not isinstance(raw_scenes, list):
        return ChapterAudioPlan()

    raw_version = value.get("version", 1)
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        version = 1

    scenes: list[AudioScenePlan] = []
    seen_scene_ids: set[str] = set()
    for scene_index, raw_scene in enumerate(raw_scenes):
        path = f"audioPlan.scenes[{scene_index}]"
        if not isinstance(raw_scene, dict):
            continue
        try:
            scene_id = _audio_string(raw_scene, "id", path)
            start_segment_index = _audio_int(
                raw_scene, "startSegmentIndex", path
            )
            end_segment_index = _audio_int(raw_scene, "endSegmentIndex", path)
        except (TypeError, ValueError):
            # A scene without a usable identity or range cannot be safely
            # placed on the narration timeline, so ignore just that scene.
            continue
        if scene_id in seen_scene_ids:
            continue
        seen_scene_ids.add(scene_id)

        music: MusicPlan | None = None
        raw_music = raw_scene.get("music")
        if isinstance(raw_music, dict):
            music_path = f"{path}.music"
            try:
                music = _parse_music_plan(raw_music, music_path)
            except (TypeError, ValueError):
                # `music` is optional.  In particular, models sometimes emit
                # a prompt string instead of the documented object; dropping
                # that asset is safer than failing the complete chapter.
                music = None

        palette: dict[str, str] = {}
        raw_palette = raw_scene.get("musicPalette")
        if isinstance(raw_palette, dict):
            palette = {
                str(key): str(item).strip()
                for key, item in raw_palette.items()
                if str(item).strip()
            }

        variants: list[MusicVariantPlan] = []
        seen_variant_ids: set[str] = set()
        raw_variants = raw_scene.get("musicVariants", [])
        if isinstance(raw_variants, list):
            for variant_index, raw_variant in enumerate(raw_variants):
                variant_path = f"{path}.musicVariants[{variant_index}]"
                try:
                    variant = _parse_music_variant(raw_variant, variant_path)
                except (TypeError, ValueError):
                    continue
                if variant.id in seen_variant_ids:
                    continue
                seen_variant_ids.add(variant.id)
                variants.append(variant)

        cues: list[MusicCuePlan] = []
        seen_cue_ids: set[str] = set()
        raw_cues = raw_scene.get("musicCues", [])
        if isinstance(raw_cues, list):
            for cue_index, raw_cue in enumerate(raw_cues):
                cue_path = f"{path}.musicCues[{cue_index}]"
                if not isinstance(raw_cue, dict):
                    continue
                try:
                    cue = MusicCuePlan(
                        id=_audio_string(raw_cue, "id", cue_path),
                        start_segment_index=_audio_int(
                            raw_cue, "startSegmentIndex", cue_path
                        ),
                        end_segment_index=_audio_int(
                            raw_cue, "endSegmentIndex", cue_path
                        ),
                        variant_id=_audio_string(raw_cue, "variantId", cue_path),
                        reason_zh=_optional_audio_string(raw_cue, "reasonZh"),
                    )
                except (TypeError, ValueError):
                    continue
                if cue.id in seen_cue_ids:
                    continue
                seen_cue_ids.add(cue.id)
                cues.append(cue)

        breaks: list[MusicBreakPlan] = []
        seen_break_indices: set[int] = set()
        raw_breaks = raw_scene.get("musicBreaks", [])
        if isinstance(raw_breaks, list):
            for break_index, raw_break in enumerate(raw_breaks):
                break_path = f"{path}.musicBreaks[{break_index}]"
                if not isinstance(raw_break, dict):
                    continue
                try:
                    item = MusicBreakPlan(
                        after_segment_index=_audio_int(
                            raw_break, "afterSegmentIndex", break_path
                        ),
                        duration_seconds=_audio_duration(raw_break, break_path),
                        reason_zh=_optional_audio_string(raw_break, "reasonZh"),
                    )
                except (TypeError, ValueError):
                    continue
                if item.after_segment_index in seen_break_indices:
                    continue
                seen_break_indices.add(item.after_segment_index)
                breaks.append(item)

        raw_sfx = raw_scene.get("sfx", [])
        if not isinstance(raw_sfx, list):
            raw_sfx = []
        sfx: list[SfxPlan] = []
        seen_sfx_ids: set[str] = set()
        for sfx_index, raw_effect in enumerate(raw_sfx):
            sfx_path = f"{path}.sfx[{sfx_index}]"
            if not isinstance(raw_effect, dict):
                continue
            try:
                effect_id = _audio_string(raw_effect, "id", sfx_path)
                effect = SfxPlan(
                    id=effect_id,
                    model=_audio_model(raw_effect, sfx_path, expected="sm-sfx"),
                    anchor_segment_index=_audio_int(
                        raw_effect, "anchorSegmentIndex", sfx_path
                    ),
                    timing=_audio_string(raw_effect, "timing", sfx_path),
                    event_zh=_optional_audio_string(raw_effect, "eventZh"),
                    duration_seconds=_audio_duration(raw_effect, sfx_path),
                    prompt=_audio_string(raw_effect, "prompt", sfx_path),
                    negative_prompt=_optional_audio_string(
                        raw_effect, "negativePrompt"
                    ),
                    reason_zh=_optional_audio_string(raw_effect, "reasonZh"),
                    anchor_text=_optional_audio_string(raw_effect, "anchorText"),
                )
            except (TypeError, ValueError):
                continue
            if effect_id in seen_sfx_ids:
                continue
            seen_sfx_ids.add(effect_id)
            sfx.append(effect)

        scenes.append(
            AudioScenePlan(
                id=scene_id,
                start_segment_index=start_segment_index,
                end_segment_index=end_segment_index,
                summary_zh=_optional_audio_string(raw_scene, "summaryZh"),
                energy_arc=_optional_audio_string(raw_scene, "energyArc"),
                music=music,
                music_palette=palette,
                music_variants=variants,
                music_cues=cues,
                music_breaks=breaks,
                sfx=sfx,
            )
        )

    if any(
        scene.music_variants or scene.music_cues or scene.music_breaks or scene.music_palette
        for scene in scenes
    ):
        version = max(version, 2)
    return ChapterAudioPlan(scenes=scenes, version=version)


def _normalize_audio_anchor_text(value: Any) -> str:
    """Normalize cue text for matching across punctuation and segment breaks."""

    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum()
    )


def _find_audio_anchor_segment(
    anchor_text: str,
    segment_texts: list[str],
    preferred_index: int,
    *,
    minimum_index: int = 0,
    maximum_index: int | None = None,
) -> tuple[int, bool] | None:
    """Find an SFX cue in normalized chapter text.

    Returns the first segment containing the cue and whether the cue begins at
    the beginning of the chapter.  Matching the concatenated normalized text
    lets a cue survive punctuation differences and a dialogue splitter that
    separated one action across adjacent segments.
    """

    needle = _normalize_audio_anchor_text(anchor_text)
    if not needle or not segment_texts:
        return None

    normalized_segments = [
        _normalize_audio_anchor_text(text) for text in segment_texts
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for text in normalized_segments:
        end = cursor + len(text)
        offsets.append((cursor, end))
        cursor = end
    haystack = "".join(normalized_segments)
    if not haystack:
        return None

    upper_bound = (
        len(segment_texts) - 1
        if maximum_index is None
        else min(maximum_index, len(segment_texts) - 1)
    )
    lower_bound = max(0, minimum_index)
    if lower_bound > upper_bound:
        return None

    candidates: list[tuple[int, int, bool]] = []
    search_from = 0
    while True:
        position = haystack.find(needle, search_from)
        if position < 0:
            break
        start_index = next(
            (
                index
                for index, (start, end) in enumerate(offsets)
                if start <= position < end
            ),
            None,
        )
        if start_index is not None and lower_bound <= start_index <= upper_bound:
            candidates.append((start_index, position, position == 0))
        search_from = position + 1

    if not candidates:
        return None
    start_index, position, starts_at_chapter_start = min(
        candidates,
        key=lambda item: (abs(item[0] - preferred_index), item[1]),
    )
    return start_index, starts_at_chapter_start


def normalize_audio_plan_anchors(
    audio_plan: ChapterAudioPlan,
    segment_texts: list[str],
) -> ChapterAudioPlan:
    """Repair model-produced SFX anchors against the actual segment text.

    This is intentionally conservative: it only changes an index when the
    supplied anchor text can be found verbatim after removing punctuation and
    whitespace.  A cue at chapter offset zero cannot occur ``before`` the
    chapter, so it is changed to ``during`` before the script is persisted.
    """

    if not segment_texts or not audio_plan.scenes:
        return audio_plan

    normalized_scenes: list[AudioScenePlan] = []
    for scene in audio_plan.scenes:
        normalized_effects: list[SfxPlan] = []
        for effect in scene.sfx:
            match = _find_audio_anchor_segment(
                effect.anchor_text,
                segment_texts,
                effect.anchor_segment_index,
                minimum_index=scene.start_segment_index,
                maximum_index=scene.end_segment_index,
            )
            if match is None:
                normalized_effects.append(effect)
                continue

            anchor_index, starts_at_chapter_start = match
            timing = (
                "during"
                if effect.timing == "before" and starts_at_chapter_start
                else effect.timing
            )
            normalized_effects.append(
                replace(
                    effect,
                    anchor_segment_index=anchor_index,
                    timing=timing,
                )
            )
        normalized_scenes.append(replace(scene, sfx=normalized_effects))
    return replace(audio_plan, scenes=normalized_scenes)


def _validate_audio_plan(
    audio_plan: ChapterAudioPlan,
    *,
    segment_count: int,
    segment_texts: list[str] | None = None,
) -> ChapterAudioPlan:
    if segment_texts is not None:
        audio_plan = normalize_audio_plan_anchors(audio_plan, segment_texts)

    # Never reject otherwise usable chapter analysis because the model placed
    # one scene or asset outside the timeline. Keep only the safe, ordered
    # subset; the planner stage adds the mandatory full-chapter music bed after
    # this defensive validation.
    valid_scenes: list[AudioScenePlan] = []
    previous_end = -1
    for scene in audio_plan.scenes:
        if not 0 <= scene.start_segment_index <= scene.end_segment_index < segment_count:
            continue
        if scene.start_segment_index <= previous_end:
            continue
        previous_end = scene.end_segment_index

        music = (
            scene.music
            if scene.music is not None and scene.music.model == "sm-music"
            else None
        )
        valid_variants: list[MusicVariantPlan] = []
        variant_ids: set[str] = set()
        for variant in scene.music_variants:
            if (
                variant.model != "sm-music"
                or variant.level not in {"low", "medium", "high"}
                or not variant.id
                or variant.id in variant_ids
            ):
                continue
            variant_ids.add(variant.id)
            valid_variants.append(variant)

        valid_cues: list[MusicCuePlan] = []
        previous_cue_end = scene.start_segment_index - 1
        cue_ids: set[str] = set()
        for cue in sorted(
            scene.music_cues,
            key=lambda item: (item.start_segment_index, item.end_segment_index),
        ):
            if (
                not cue.id
                or cue.id in cue_ids
                or cue.variant_id not in variant_ids
                or not scene.start_segment_index
                <= cue.start_segment_index
                <= cue.end_segment_index
                <= scene.end_segment_index
                or cue.start_segment_index <= previous_cue_end
            ):
                continue
            cue_ids.add(cue.id)
            previous_cue_end = cue.end_segment_index
            valid_cues.append(cue)

        valid_breaks: list[MusicBreakPlan] = []
        break_indices: set[int] = set()
        # Keep one deliberate breathing window per scene. More windows make a
        # long bed feel chopped up and are better represented as cue changes.
        for item in sorted(
            scene.music_breaks,
            key=lambda value: value.after_segment_index,
        ):
            if (
                item.after_segment_index in break_indices
                or not scene.start_segment_index
                <= item.after_segment_index
                < scene.end_segment_index
                or not 2.0 <= item.duration_seconds <= 6.0
                or valid_breaks
            ):
                continue
            break_indices.add(item.after_segment_index)
            valid_breaks.append(item)

        valid_effects: list[SfxPlan] = []
        for effect in scene.sfx:
            if (
                effect.model != "sm-sfx"
                or effect.timing not in _AUDIO_TIMINGS
                or not scene.start_segment_index
                <= effect.anchor_segment_index
                <= scene.end_segment_index
            ):
                continue
            valid_effects.append(effect)
        valid_scenes.append(
            replace(
                scene,
                music=music,
                music_variants=valid_variants,
                music_cues=valid_cues,
                music_breaks=valid_breaks,
                sfx=valid_effects,
            )
        )

    return replace(audio_plan, scenes=valid_scenes)


_DEFAULT_CONTINUOUS_MUSIC = {
    "zh": MusicPlan(
        model="sm-music",
        duration_seconds=30.0,
        prompt=(
            "TrackType: Music, VocalType: Instrumental, Genre: historical ambient, "
            "Instruments: guqin, xiao, low strings, 68 BPM, restrained nocturnal "
            "suspense, sparse background texture"
        ),
        negative_prompt=(
            "speech, vocals, lyrics, sound effects, abrupt hits"
        ),
        reason_zh="确保整章文本始终有连续、克制且不遮挡对白的背景音乐。",
    ),
    "en": MusicPlan(
        model="sm-music",
        duration_seconds=30.0,
        prompt=(
            "TrackType: Music, VocalType: Instrumental, Genre: historical ambient, "
            "Instruments: guqin, xiao, low strings, 68 BPM, restrained nocturnal "
            "suspense, sparse background texture"
        ),
        negative_prompt=(
            "speech, vocals, lyrics, sound effects, abrupt hits"
        ),
        reason_zh="Keep a continuous restrained music bed under the full chapter.",
    ),
}


def _ensure_variant_audio_music_coverage(
    audio_plan: ChapterAudioPlan,
    *,
    segment_count: int,
    language: str,
) -> ChapterAudioPlan:
    """Normalize v2 scenes without erasing intentional music breaks."""

    if segment_count <= 0:
        return audio_plan

    fallback = _DEFAULT_CONTINUOUS_MUSIC[
        "zh" if str(language).lower().startswith("zh") else "en"
    ]
    scenes = list(audio_plan.scenes)
    normalized: list[AudioScenePlan] = []
    used_ids = {scene.id for scene in scenes}
    cursor = 0
    fill_number = 1

    def append_fill(start: int, end: int) -> None:
        nonlocal fill_number
        if start > end:
            return
        scene_id = f"music_fill_{fill_number:03d}"
        while scene_id in used_ids:
            fill_number += 1
            scene_id = f"music_fill_{fill_number:03d}"
        used_ids.add(scene_id)
        fill_music = replace(
            fallback,
            prompt=_compact_music_prompt(
                f"{fallback.prompt}, sparse neutral transition, section {fill_number}"
            ),
            reason_zh="补齐新版音乐计划缺失的场景范围。",
        )
        normalized.append(
            AudioScenePlan(
                id=scene_id,
                start_segment_index=start,
                end_segment_index=end,
                summary_zh="补齐背景音乐覆盖",
                music=fill_music,
                sfx=[],
            )
        )
        fill_number += 1

    for scene in scenes:
        start = max(0, scene.start_segment_index)
        end = min(segment_count - 1, scene.end_segment_index)
        if end < cursor:
            continue
        if start > cursor:
            append_fill(cursor, start - 1)
        start = max(start, cursor)

        variants = list(scene.music_variants)
        cues = [
            cue
            for cue in scene.music_cues
            if start <= cue.start_segment_index <= cue.end_segment_index <= end
            and any(variant.id == cue.variant_id for variant in variants)
        ]
        if variants and not cues:
            preferred = next(
                (variant for variant in variants if variant.level == "low"),
                variants[0],
            )
            cues = [
                MusicCuePlan(
                    id=f"{scene.id}_cue_001",
                    start_segment_index=start,
                    end_segment_index=end,
                    variant_id=preferred.id,
                    reason_zh="没有可用变体调度时使用低能量同主题铺底。",
                )
            ]
        normalized.append(
            replace(
                scene,
                start_segment_index=start,
                end_segment_index=end,
                # v2 variants are the canonical assets; do not generate a
                # duplicate legacy scene asset when a model emitted both.
                music=None if variants else scene.music,
                music_variants=variants,
                music_cues=cues,
                music_breaks=[
                    item
                    for item in scene.music_breaks
                    if start <= item.after_segment_index < end
                ],
                sfx=[
                    effect
                    for effect in scene.sfx
                    if start <= effect.anchor_segment_index <= end
                ],
            )
        )
        cursor = end + 1

    if cursor < segment_count:
        append_fill(cursor, segment_count - 1)
    if not normalized:
        append_fill(0, segment_count - 1)
    return ChapterAudioPlan(scenes=normalized, version=max(2, audio_plan.version))


def ensure_audio_music_coverage(
    audio_plan: ChapterAudioPlan,
    *,
    segment_count: int,
    language: str = "zh",
    segments: list[dict[str, Any]] | None = None,
) -> ChapterAudioPlan:
    """Guarantee continuous, scene-aware music over the complete timeline.

    The planner must not leave dialogue-only gaps, but preserving every
    model-produced music scene is important: collapsing scenes into one asset
    makes a long chapter monotonous. This function only fills missing ranges
    or missing music objects. A model-produced prompt is preferred; the
    built-in prompt is a safety net for malformed or incomplete output.
    """

    if segment_count <= 0:
        return audio_plan

    if any(
        scene.music_variants or scene.music_cues or scene.music_breaks
        for scene in audio_plan.scenes
    ):
        return _ensure_variant_audio_music_coverage(
            audio_plan,
            segment_count=segment_count,
            language=language,
        )

    source_segments = segments or []
    scenes = list(audio_plan.scenes)
    source_music = next(
        (scene for scene in scenes if scene.music is not None),
        None,
    )
    music = (
        source_music.music
        if source_music is not None and source_music.music is not None
        else _DEFAULT_CONTINUOUS_MUSIC[
            "zh" if str(language).lower().startswith("zh") else "en"
        ]
    )

    def section_music(
        start: int,
        end: int,
        section_number: int,
        base_music: MusicPlan | None = None,
    ) -> MusicPlan:
        base = base_music or music
        text = " ".join(
            str(item.get("text") or "")
            for item in source_segments[start : min(end + 1, len(source_segments))]
            if isinstance(item, dict)
        )
        emotions = {
            str(item.get("emotion") or "neutral").strip().lower()
            for item in source_segments[start : min(end + 1, len(source_segments))]
            if isinstance(item, dict)
        }
        if emotions & {"tense", "afraid", "nervous", "angry", "contemptuous"}:
            variation = "darker suspense, restrained pulse"
        elif emotions & {"sad", "grief", "tired", "solemn", "bitter"}:
            variation = "somber reflection, slow sustained tones"
        elif emotions & {"happy", "excited", "teasing", "gentle"}:
            variation = "warm uplift, light rhythmic motion"
        elif any(keyword in text for keyword in ("火堆", "营地", "帐篷", "夜")):
            variation = "warm night ambience, gentle plucked motion"
        elif any(keyword in text for keyword in ("仇家", "宗", "危险", "追", "伤")):
            variation = "watchful tension, muted low strings"
        else:
            variation = "curious reflection, gentle motion"
        phase_palette = (
            "low strings and muted woodwind",
            "warm plucked strings and soft woodwind",
            "bamboo flute and high strings",
        )[(section_number - 1) % 3]
        prompt = _compact_music_prompt(
            f"{base.prompt}, {variation}, {phase_palette}, section {section_number}"
        )
        reason = base.reason_zh or "补齐连续背景音乐覆盖。"
        return replace(
            base,
            prompt=prompt,
            reason_zh=f"{reason} 根据第 {start}–{end} 段文本与情绪补充段落变化。",
        )

    if not scenes:
        scenes = [
            AudioScenePlan(
                id="chapter_music",
                start_segment_index=0,
                end_segment_index=segment_count - 1,
                summary_zh="整章连续背景音乐",
                music=section_music(0, segment_count - 1, 1),
                sfx=[],
            )
        ]

    normalized: list[AudioScenePlan] = []
    used_ids = {scene.id for scene in scenes}
    cursor = 0
    fill_number = 1

    def append_fill(start: int, end: int) -> None:
        nonlocal fill_number
        if start > end:
            return
        scene_id = f"music_fill_{fill_number:03d}"
        while scene_id in used_ids:
            fill_number += 1
            scene_id = f"music_fill_{fill_number:03d}"
        used_ids.add(scene_id)
        normalized.append(
            AudioScenePlan(
                id=scene_id,
                start_segment_index=start,
                end_segment_index=end,
                summary_zh="补齐连续背景音乐覆盖",
                music=section_music(start, end, fill_number),
                sfx=[],
            )
        )
        fill_number += 1

    for scene_number, scene in enumerate(scenes, start=1):
        start = max(0, scene.start_segment_index)
        end = min(segment_count - 1, scene.end_segment_index)
        if end < cursor:
            continue
        if start > cursor:
            append_fill(cursor, start - 1)
        start = max(start, cursor)
        scene_music = scene.music or section_music(start, end, scene_number)
        scene_sfx = [
            effect
            for effect in scene.sfx
            if start <= effect.anchor_segment_index <= end
        ]
        normalized.append(
            replace(
                scene,
                start_segment_index=start,
                end_segment_index=end,
                music=scene_music,
                sfx=scene_sfx,
            )
        )
        cursor = end + 1

    if cursor < segment_count:
        append_fill(cursor, segment_count - 1)

    if not normalized:
        normalized.append(
            AudioScenePlan(
                id="chapter_music",
                start_segment_index=0,
                end_segment_index=segment_count - 1,
                summary_zh="整章连续背景音乐",
                music=section_music(0, segment_count - 1, 1),
                sfx=[],
            )
        )

    def phase_count(start: int, end: int) -> int:
        length = end - start + 1
        if length < 45:
            return 1
        count = max(1, math.ceil(length / 40))
        # A chapter around the size of a normal long audiobook scene should
        # not fall back to one chapter-wide bed just because the planner
        # returned one broad scene.  Three phases is the minimum useful
        # amount of change for a 90+ segment chapter.
        if segment_count >= 90 and length >= 80:
            count = max(3, count)
        return count

    expanded: list[AudioScenePlan] = []
    for scene_number, scene in enumerate(normalized, start=1):
        count = phase_count(scene.start_segment_index, scene.end_segment_index)
        if count == 1:
            expanded.append(scene)
            continue

        start = scene.start_segment_index
        length = scene.end_segment_index - start + 1
        for phase in range(count):
            phase_start = start + (length * phase) // count
            phase_end = start + (length * (phase + 1)) // count - 1
            phase_id = f"{scene.id}_phase_{phase + 1:03d}"
            phase_effects = [
                effect
                for effect in scene.sfx
                if phase_start <= effect.anchor_segment_index <= phase_end
            ]
            expanded.append(
                replace(
                    scene,
                    id=phase_id,
                    start_segment_index=phase_start,
                    end_segment_index=phase_end,
                    summary_zh=(
                        f"{scene.summary_zh}（音乐阶段 {phase + 1}/{count}）"
                    ).strip("（）"),
                    music=section_music(
                        phase_start,
                        phase_end,
                        scene_number * 100 + phase + 1,
                        scene.music,
                    ),
                    sfx=phase_effects,
                )
            )

    return ChapterAudioPlan(scenes=expanded)


def audio_plan_to_dict(audio_plan: ChapterAudioPlan) -> dict[str, Any]:
    """Serialize the analysis audio plan into the chapter-script JSON shape."""
    serialized_scenes: list[dict[str, Any]] = []
    has_v2_fields = audio_plan.version >= 2
    for scene in audio_plan.scenes:
        item: dict[str, Any] = {
            "id": scene.id,
            "startSegmentIndex": scene.start_segment_index,
            "endSegmentIndex": scene.end_segment_index,
            "summaryZh": scene.summary_zh,
            "music": (
                {
                    "model": scene.music.model,
                    "durationSeconds": scene.music.duration_seconds,
                    "prompt": scene.music.prompt,
                    "negativePrompt": scene.music.negative_prompt,
                    "reasonZh": scene.music.reason_zh,
                }
                if scene.music is not None
                else None
            ),
            "sfx": [
                {
                    "id": effect.id,
                    "model": effect.model,
                    "anchorSegmentIndex": effect.anchor_segment_index,
                    "anchorText": effect.anchor_text,
                    "timing": effect.timing,
                    "eventZh": effect.event_zh,
                    "durationSeconds": effect.duration_seconds,
                    "prompt": effect.prompt,
                    "negativePrompt": effect.negative_prompt,
                    "reasonZh": effect.reason_zh,
                }
                for effect in scene.sfx
            ],
        }
        if scene.energy_arc:
            item["energyArc"] = scene.energy_arc
        if (
            scene.music_palette
            or scene.music_variants
            or scene.music_cues
            or scene.music_breaks
        ):
            has_v2_fields = True
            item["musicPalette"] = dict(scene.music_palette)
            item["musicVariants"] = [
                {
                    "id": variant.id,
                    "level": variant.level,
                    "model": variant.model,
                    "durationSeconds": variant.duration_seconds,
                    "prompt": variant.prompt,
                    "negativePrompt": variant.negative_prompt,
                    "reasonZh": variant.reason_zh,
                }
                for variant in scene.music_variants
            ]
            item["musicCues"] = [
                {
                    "id": cue.id,
                    "startSegmentIndex": cue.start_segment_index,
                    "endSegmentIndex": cue.end_segment_index,
                    "variantId": cue.variant_id,
                    "reasonZh": cue.reason_zh,
                }
                for cue in scene.music_cues
            ]
            item["musicBreaks"] = [
                {
                    "afterSegmentIndex": item.after_segment_index,
                    "durationSeconds": item.duration_seconds,
                    "reasonZh": item.reason_zh,
                }
                for item in scene.music_breaks
            ]
        serialized_scenes.append(item)

    payload: dict[str, Any] = {"scenes": serialized_scenes}
    if has_v2_fields:
        payload["version"] = max(2, audio_plan.version)
    # The planner may restart SFX/variant numbering in each scene. Normalize
    # at the serialization boundary so every newly persisted chapter plan is
    # safe for the chapter-wide Stable Audio manifest.
    normalize_audio_plan_asset_ids(payload)
    return payload


def _parse_analysis_json(payload: dict[str, Any]) -> ChapterAnalysisResult:
    characters = [
        CharacterAnalysis(
            id=str(item["id"]),
            canonical_name=str(item["canonicalName"]),
            aliases=[str(alias) for alias in item.get("aliases", [])],
            gender=str(item.get("gender", "unknown")),
            age_class=str(item.get("ageClass", "unknown")),
            confidence=float(item.get("confidence", 0.0)),
            voice_design=str(item.get("voiceDesign", "") or "").strip(),
        )
        for item in payload.get("characters", [])
    ]
    annotations = [
        SegmentAnnotation(
            segment_index=int(item["segmentIndex"]),
            speaker_id=str(item.get("speakerId", "unknown")),
            emotion=_normalize_emotion(item.get("emotion", "neutral")),
            confidence=float(item.get("confidence", 0.0)),
            pace=_normalize_pace(item.get("pace", "normal")),
            warnings=[str(warning) for warning in item.get("warnings", [])],
        )
        for item in payload.get("segmentAnnotations", [])
    ]
    return ChapterAnalysisResult(
        characters=characters,
        segment_annotations=annotations,
        audio_plan=_parse_audio_plan(payload.get("audioPlan")),
    )


def _normalize_and_validate_analysis(
    result: ChapterAnalysisResult,
    *,
    request: ChapterAnalysisRequest,
    segment_count: int,
    segment_texts: list[str] | None = None,
) -> ChapterAnalysisResult:
    indices = [annotation.segment_index for annotation in result.segment_annotations]
    expected_indices = list(range(segment_count))
    if sorted(indices) != expected_indices or len(indices) != len(set(indices)):
        raise ValueError(
            "invalid segment annotations: expected each segment index exactly once "
            f"({expected_indices}), got {indices}"
        )

    known_by_name: dict[str, CharacterContext] = {}
    for known in request.known_characters:
        for name in [known.id, known.canonical_name, *known.aliases]:
            normalized_name = name.strip().casefold()
            if normalized_name:
                known_by_name[normalized_name] = known

    id_replacements: dict[str, str] = {}
    normalized_characters: list[CharacterAnalysis] = []
    seen_character_ids: set[str] = set()
    for character in result.characters:
        if character.id.strip().casefold() == "narrator":
            continue
        known = None
        for name in [character.id, character.canonical_name, *character.aliases]:
            known = known_by_name.get(name.strip().casefold())
            if known is not None:
                break

        if known is None:
            normalized = character
        else:
            id_replacements[character.id] = known.id
            aliases = sorted(
                {
                    *known.aliases,
                    *character.aliases,
                    character.canonical_name,
                }
                - {known.canonical_name}
            )
            normalized = CharacterAnalysis(
                id=known.id,
                canonical_name=known.canonical_name,
                aliases=aliases,
                gender=(
                    known.gender
                    if known.gender != "unknown"
                    else character.gender
                ),
                age_class=(
                    known.age_class
                    if known.age_class != "unknown"
                    and character.age_class == "unknown"
                    else character.age_class
                ),
                confidence=character.confidence,
                voice_design=character.voice_design or known.voice_design,
            )

        if normalized.id not in seen_character_ids:
            normalized_characters.append(normalized)
            seen_character_ids.add(normalized.id)

    normalized_annotations = [
        SegmentAnnotation(
            segment_index=annotation.segment_index,
            speaker_id=id_replacements.get(
                annotation.speaker_id,
                annotation.speaker_id,
            ),
            emotion=annotation.emotion,
            confidence=annotation.confidence,
            pace=annotation.pace,
            warnings=annotation.warnings,
        )
        for annotation in result.segment_annotations
    ]

    valid_speaker_ids = {
        "narrator",
        "unknown",
        *(character.id for character in normalized_characters),
        *(known.id for known in request.known_characters),
    }
    invalid_speakers = sorted(
        {
            annotation.speaker_id
            for annotation in normalized_annotations
            if annotation.speaker_id not in valid_speaker_ids
        }
    )
    if invalid_speakers:
        raise ValueError(
            f"unknown speakerId references: {', '.join(invalid_speakers)}"
        )

    audio_plan = _validate_audio_plan(
        result.audio_plan,
        segment_count=segment_count,
        segment_texts=segment_texts,
    )

    return ChapterAnalysisResult(
        characters=normalized_characters,
        segment_annotations=normalized_annotations,
        audio_plan=audio_plan,
        voice_directions=result.voice_directions,
    )


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {error.code}: {detail}") from error


def _guess_gender(name: str) -> str:
    if name.lower() in {"elizabeth", "jane", "mary", "anna", "emma"}:
        return "female"
    if name.lower() in {"darcy", "john", "william", "charles"}:
        return "male"
    return "unknown"


def _guess_emotion(text: str) -> str:
    lowered = text.casefold()
    if any(
        marker in lowered
        for marker in (
            "teasing",
            "mocking",
            "sarcastic",
            "taunting",
            "sneered",
            "戏谑",
            "嘲弄",
            "嘲讽",
            "讥讽",
            "挖苦",
            "奚落",
            "取笑",
            "轻蔑",
            "哟",
            "啧",
            "还挺",
            "废物",
            "喂狗",
            "逗狗",
            "孝敬师兄",
            "正合你用",
        )
    ):
        return "teasing"
    if any(word in lowered for word in ["whispered", "murmured", "breathed"]):
        return "whispering"
    if any(word in lowered for word in ["afraid", "scared", "terrified", "fear"]):
        return "afraid"
    if any(word in lowered for word in ["sobbed", "cried", "wept", "tearfully"]):
        return "sad"
    if any(word in lowered for word in ["shouted", "cried out", "exclaimed", "snapped"]):
        return "angry"
    if "!" in text:
        return "excited"
    return "neutral"


def _normalize_emotion(value: Any) -> str:
    emotion = str(value or "neutral").strip().casefold()
    aliases = {
        "mocking": "teasing",
        "sarcastic": "teasing",
        "taunting": "teasing",
        "嘲弄": "teasing",
        "嘲讽": "teasing",
        "戏谑": "teasing",
    }
    emotion = aliases.get(emotion, emotion)
    return (
        emotion
        if emotion in {
            "neutral",
            "happy",
            "sad",
            "angry",
            "afraid",
            "tense",
            "teasing",
            "whispering",
            "excited",
            "tired",
            "grief",
            "cold",
            "pleading",
            "surprised",
            "gentle",
            "resolute",
            "nervous",
            "contemptuous",
            "solemn",
            "bitter",
        }
        else "neutral"
    )


def _normalize_pace(value: Any) -> str:
    pace = str(value or "normal").strip().lower()
    return pace if pace in {"slow", "normal", "fast"} else "normal"
