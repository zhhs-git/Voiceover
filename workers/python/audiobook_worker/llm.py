from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable
from urllib import request as urllib_request
from urllib.error import HTTPError

from audiobook_worker.dialogue import resolve_text_language, segment_dialogue


@dataclass(frozen=True)
class CharacterContext:
    """A character already identified in a previous chapter, passed for consistency."""
    id: str
    canonical_name: str
    aliases: list[str]
    gender: str
    age_class: str = "unknown"


@dataclass(frozen=True)
class ChapterAnalysisRequest:
    book_id: str
    chapter_id: str
    text: str
    language: str
    known_characters: list[CharacterContext] = field(default_factory=list)


@dataclass(frozen=True)
class CharacterAnalysis:
    id: str
    canonical_name: str
    aliases: list[str]
    gender: str
    age_class: str
    confidence: float


@dataclass(frozen=True)
class SegmentAnnotation:
    segment_index: int
    speaker_id: str
    emotion: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    pace: str = "normal"


@dataclass(frozen=True)
class ChapterAnalysisResult:
    characters: list[CharacterAnalysis]
    segment_annotations: list[SegmentAnnotation]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3


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
- `pace`: "slow" | "normal" | "fast". For narration, always return "normal". For \
  dialogue, apply the mandatory pace rubric below; do not mechanically use the same \
  pace for every speaker.
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
pace 只能是 slow、normal、fast。speakerId="narrator" 的旁白必须返回 normal。语速只表示说话的快慢节奏，不表示音量、音色、年龄、性别、身份地位或声音本身特质；不能因为内容严肃、句子较长、人物年长或身份威严就直接判为 slow。

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
旁白（narrator）必须返回 normal，无例外。茶肆/街头/集市匿名闲谈：普通聊天用 normal，短促连续问答用 fast，只有明确低声警告或犹豫才用 slow，不能因一句警告将整场对白改为 slow。

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


@dataclass(frozen=True)
class ResolvedModel:
    provider: str
    model_id: str
    base_url: str
    api_key: str
    api: str
    family: str
    max_tokens: int


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
        request = replace(
            request,
            language=resolve_text_language(request.text, request.language),
        )
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _system_prompt(request.language)},
                {"role": "user", "content": _analysis_prompt(request)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

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
                result = _parse_analysis_json(json.loads(content))
                segment_count = len(
                    segment_dialogue(request.text, language=request.language)
                )
                return _normalize_and_validate_analysis(
                    result,
                    request=request,
                    segment_count=segment_count,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    time.sleep(2 ** attempt)
                    continue

        raise RuntimeError(
            f"LLM analysis failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error


def default_analyzer():
    model_override = os.environ.get("AUDIOBOOK_LLM_MODEL")
    if model_override == "mock":
        return MockLLMAnalyzer()
    resolved = resolve_model(model_override)
    analyzer = analyzer_from_resolved_model(resolved) if resolved else None
    if analyzer is not None:
        return analyzer
    return MockLLMAnalyzer()


def resolve_model(model_arg: str | None = None) -> ResolvedModel | None:
    config = read_models_json()
    if config is not None:
        return resolve_model_from_config(config, model_arg)

    base_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return None
    return ResolvedModel(
        provider="env",
        model_id=os.environ.get("MODEL_ID") or os.environ.get("OPENAI_MODEL") or "gpt-4o",
        base_url=base_url,
        api_key=os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "unused",
        api="openai-completions",
        family="default",
        max_tokens=8192,
    )


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


def _parse_analysis_json(payload: dict[str, Any]) -> ChapterAnalysisResult:
    characters = [
        CharacterAnalysis(
            id=str(item["id"]),
            canonical_name=str(item["canonicalName"]),
            aliases=[str(alias) for alias in item.get("aliases", [])],
            gender=str(item.get("gender", "unknown")),
            age_class=str(item.get("ageClass", "unknown")),
            confidence=float(item.get("confidence", 0.0)),
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
    return ChapterAnalysisResult(characters=characters, segment_annotations=annotations)


def _normalize_and_validate_analysis(
    result: ChapterAnalysisResult,
    *,
    request: ChapterAnalysisRequest,
    segment_count: int,
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

    return ChapterAnalysisResult(
        characters=normalized_characters,
        segment_annotations=normalized_annotations,
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
