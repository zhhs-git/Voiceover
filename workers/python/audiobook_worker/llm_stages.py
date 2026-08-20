"""Independent LLM stages used by the audiobook pipeline.

The original analyzer asked one model response to discover characters, assign
speakers, judge delivery, and design Stable Audio assets at the same time.
This module keeps those responsibilities separate.  The caller still receives
the existing ChapterAnalysisResult shape so the rest of the worker remains
backwards compatible.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

from audiobook_worker.dialogue import resolve_text_language, segment_dialogue
from audiobook_worker.llm import (
    AudioPlanningRequest,
    AudioScenePlan,
    ChapterAnalysisRequest,
    ChapterAnalysisResult,
    CharacterAnalysis,
    CharacterContext,
    ChapterAudioPlan,
    SegmentAnnotation,
    _CHINESE_SYSTEM_PROMPT,
    _ENGLISH_SYSTEM_PROMPT,
    _normalize_and_validate_analysis,
    _normalize_emotion,
    _normalize_pace,
    _parse_audio_plan,
    _validate_audio_plan,
    ensure_audio_music_coverage,
    select_active_audio_characters,
)


def _legacy_prompt_section(
    prompt: str,
    heading: str,
    next_heading: str | None = None,
) -> str:
    """Reuse a focused section of the original, battle-tested analyzer prompt.

    The old prompt contains more responsibilities than any one split stage
    should receive.  Extracting only the matching sections keeps the new calls
    narrow while ensuring that a rule is not silently lost during the split.
    """

    start = prompt.find(heading)
    if start < 0:
        raise RuntimeError(f"legacy prompt section not found: {heading}")
    end = len(prompt)
    if next_heading:
        candidate = prompt.find(next_heading, start + len(heading))
        if candidate >= 0:
            end = candidate
    return prompt[start:end].strip()


_LEGACY_ROLE_GUIDANCE = {
    "zh": _legacy_prompt_section(
        _CHINESE_SYSTEM_PROMPT,
        "## 角色规则",
        "## 情绪、语速、置信度与警告",
    ),
    "en": _legacy_prompt_section(
        _ENGLISH_SYSTEM_PROMPT,
        "## Rules for characters",
        "## Speaker attribution rules",
    ),
}

_LEGACY_IDENTITY_GUIDANCE = {
    "zh": "\n\n".join(
        (
            _legacy_prompt_section(
                _CHINESE_SYSTEM_PROMPT,
                "## 跨章节角色身份规则（最高优先级）",
                "## 总体原则",
            ),
            _LEGACY_ROLE_GUIDANCE["zh"],
        )
    ),
    "en": "\n\n".join(
        (
            _legacy_prompt_section(
                _ENGLISH_SYSTEM_PROMPT,
                "## Cross-chapter identity rules (highest priority)",
                "## Rules for characters",
            ),
            _LEGACY_ROLE_GUIDANCE["en"],
        )
    ),
}

_LEGACY_SPEAKER_GUIDANCE = {
    "zh": "\n\n".join(
        (
            _legacy_prompt_section(
                _CHINESE_SYSTEM_PROMPT,
                "## 总体原则",
                "## 中文说话人归属规则",
            ),
            _legacy_prompt_section(
                _CHINESE_SYSTEM_PROMPT,
                "## 中文说话人归属规则",
                "## 角色规则",
            ),
            _legacy_prompt_section(
                _CHINESE_SYSTEM_PROMPT,
                "## 最终自检",
                "## 示例",
            ),
            _legacy_prompt_section(
                _CHINESE_SYSTEM_PROMPT,
                "## 示例",
                "## 输出格式",
            ),
        )
    ),
    "en": "\n\n".join(
        (
            _legacy_prompt_section(
                _ENGLISH_SYSTEM_PROMPT,
                "## Speaker attribution rules",
                "## Rules for segmentAnnotations",
            ),
            _legacy_prompt_section(
                _ENGLISH_SYSTEM_PROMPT,
                "## Final self-check",
                "## Output format",
            ),
        )
    ),
}

_LEGACY_DELIVERY_GUIDANCE = {
    "zh": _legacy_prompt_section(
        _CHINESE_SYSTEM_PROMPT,
        "## 情绪、语速、置信度与警告",
        "## 最终自检",
    ),
    "en": _legacy_prompt_section(
        _ENGLISH_SYSTEM_PROMPT,
        "## Rules for segmentAnnotations",
        "## Final self-check",
    ),
}


_CHARACTER_PROMPTS = {
    "zh": (
        "你是有声书的角色身份分析器。你只负责识别本章中真正需要独立配音的角色，并与全书角色名册进行身份消歧。不要判断每个片段是谁说的，不要分析情绪、语速、场景、背景音乐或音效。"
        "\n\n通读完整章节、所有预切分片段和 knownCharacters；先建立场景和人物关系，再识别实际发言或内心独白且需要独立配音的角色。不得把地点、组织、书名、物品、章节标题、被提及但没有发言的人，或被引用的称谓、名字、标签、诗句、公告、转述和音效词登记为角色。"
        + "\n\n"
        + _LEGACY_IDENTITY_GUIDANCE["zh"]
        + "\n\n本阶段专属边界：只返回本章活跃角色，不返回 segmentAnnotations、emotion、pace、voiceDesign、direction、audioPlan 或其它字段。角色候选 ID 可以是本次响应内的临时键，但已知角色必须使用名册中的精确 ID。"
        + "\n\n只返回 JSON：{\"characters\":[{\"id\":\"候选ID\",\"canonicalName\":\"角色名\",\"aliases\":[],\"gender\":\"male\",\"ageClass\":\"adult\",\"confidence\":0.0}]}。"
    ),
    "en": (
        "You are the character-identity stage of an audiobook pipeline. Identify only the real characters who need independent voices in this chapter and resolve them against the book roster. Do not assign speakers to segments or judge delivery, scenes, music, or sound effects."
        "\n\nRead the complete chapter, all pre-segmented text, and knownCharacters; establish scene and relationship context before identifying characters who actually speak or have an independently voiced inner monologue. Do not register places, organizations, books, objects, chapter titles, mentioned non-speakers, or quoted names, labels, poems, announcements, reported speech, or sound effects as characters."
        + "\n\n"
        + _LEGACY_IDENTITY_GUIDANCE["en"]
        + "\n\nStage boundary: return only active chapter characters. Do not return segmentAnnotations, emotion, pace, voiceDesign, direction, audioPlan, or any other field. Temporary candidate IDs are allowed only for newly discovered characters; known characters must use their exact roster IDs."
        + "\n\nReturn only JSON: {\"characters\":[{\"id\":\"candidate_id\",\"canonicalName\":\"name\",\"aliases\":[],\"gender\":\"male\",\"ageClass\":\"adult\",\"confidence\":0.0}]}."
    ),
}

_VOICE_DESIGN_PROMPTS = {
    "zh": (
        "你是有声书的角色固定音色设计器。这是一个独立的角色音色阶段，不负责角色识别、说话人归属、情绪标注或当前场景导演。请根据完整章节、当前章节实际发言角色、全书角色名册和已有音色画像，为每个当前活跃角色生成跨章节稳定、可被不同语音合成后端理解的自然语言音色画像。"
        + "\n\n"
        + _LEGACY_ROLE_GUIDANCE["zh"]
        + "\n\n"
        + "以上旧角色规则只用于继承身份、称谓、性别、年龄和角色一致性约束；本阶段不得重新识别角色或创建角色。\n\n"
        + "要求：\n1. 已有角色如果已经有 voiceDesign，必须原样保留，不要因为本章场景而改写；只有缺失或为空时才生成新的设计。\n2. 新设计必须结合正文中明确的身份、性格底色、外形气质、人物关系和说话习惯，不能只写性别和年龄。\n3. 角色之间要形成可听见的差异：音高/音域、声音厚薄、共鸣位置、气息强弱、咬字方式、停顿习惯、语尾和情绪外显程度至少有几项明确不同。\n4. 设计必须是稳定的基础声线，不写当前场景、临时情绪、背景音乐、音效、混响、EQ、压缩或其它后期处理。\n5. 使用简洁、具体、生动且无矛盾的自然语言，输出 1–4 句，不使用“普通”“正常”等模糊词，也不要写任何模型、接口或提示词语法。\n6. 角色必须使用输入中的精确 id。"
        + "\n\nvoiceDesign 建议包含：角色身份与性格底色；音色质感、音域和共鸣；长期说话习惯；与其他角色区分及禁止混淆的固定约束。只返回 JSON：{\"characters\":[{\"id\":\"角色精确ID\",\"voiceDesign\":\"1–4句稳定的中文音色设计\"}]}。不要返回 speakerId、segmentAnnotations、emotion、pace、direction、audioPlan 或其它字段。"
    ),
    "en": (
        "You are the independent fixed voice-design stage for an audiobook. Do not identify characters, assign speakers, judge delivery, direct the current scene, or plan music and sound effects. Based on the complete chapter, active speaking characters, the book roster, and any existing voice designs, create a stable natural-language voice profile that can be understood by different speech-synthesis backends for each active character."
        + "\n\n"
        + _LEGACY_ROLE_GUIDANCE["en"]
        + "\n\nThe legacy character rules above are inherited only for identity, names, aliases, gender, age, and cross-chapter consistency; this stage must not rediscover or create characters.\n\n"
        + "Rules:\n1. If a known character already has `voiceDesign`, preserve it verbatim. Only create a new design when it is missing or empty; do not rewrite a stable voice for a chapter-specific situation.\n2. Use identity, temperament, relationships, appearance cues, and speech habits from the text. Do not reduce the design to gender and age.\n3. Make characters audibly distinct through several concrete anchors: pitch/range, weight, resonance, breath, diction, pauses, sentence endings, and emotional transparency.\n4. Describe only the stable baseline voice. Do not include the current scene, temporary emotion, music, SFX, reverb, EQ, compression, post-processing, model names, or prompt syntax.\n5. Use concise, concrete, vivid natural language with no contradictory or vague traits. Keep each design to 1–4 sentences.\n6. Use the exact input character id.\n\nReturn only JSON: {\"characters\":[{\"id\":\"exact_character_id\",\"voiceDesign\":\"1–4 sentences describing the stable voice\"}]}. Do not return speakerId, segmentAnnotations, emotion, pace, direction, audioPlan, or any other field."
    ),
}

_VOICE_DIRECTION_PROMPTS = {
    "zh": (
        "你是独立的有声书动态演绎导演。角色身份、固定音色、说话人、情绪和粗粒度语速已经由其它阶段确定。你只为每个文本片段生成供 MiMo V2.5 使用的自然语言演绎指导，不修改角色、不修改 speakerId、不重新设计固定音色，也不规划背景音乐或音效。"
        + "\n\n"
        + _LEGACY_DELIVERY_GUIDANCE["zh"]
        + "\n\n请结合当前片段、前后邻近片段、人物关系、当前 emotion 和粗粒度 pace，具体描述：语速与变化趋势、节奏是否从容或急促、停顿位置与长短、气息、重音、咬字力度、句尾处理和情绪起伏。不要只输出“慢/正常/快”，要写成可执行的导演指令，例如“中等偏快但保持从容”“慢速且有压迫感”“断续迟疑，停顿偏多”“语速逐渐加快，句间停顿缩短”。"
        + "\n\n不要把严肃内容自动判为慢速；不要把 whispering 自动判为慢速；如果文本没有明确速度证据，仍需根据人物固定习惯、句子节奏和情绪给出自然、克制的指导。每个输入目标片段必须返回一次。此阶段只返回 directions，不得返回 characters、voiceDesign、speakerId、emotion、pace、audioPlan 或其它字段。只返回 JSON：{\"directions\":[{\"segmentIndex\":0,\"direction\":\"细粒度的语速、节奏、停顿、气息、重音和情绪指导\"}]}。"
    ),
    "en": (
        "You are the independent dynamic performance-directing stage for an audiobook. Character identity, fixed voice design, speaker attribution, emotion, and coarse pace have already been decided by other stages. Generate only natural-language performance directions for MiMo V2.5 for every target segment. Do not change characters or speakerId, redesign the fixed voice, or plan music and sound effects."
        + "\n\n"
        + _LEGACY_DELIVERY_GUIDANCE["en"]
        + "\n\nUse the current segment, nearby context, character relationships, emotion, and coarse pace. Describe actionable speed and tempo, whether the delivery is composed or hurried, pause placement and length, breath, emphasis, diction, sentence endings, and emotional movement. Do not output only slow/normal/fast; write a usable direction such as “moderately fast but composed”, “slow with pressure”, “hesitant and broken with frequent pauses”, or “gradually accelerate while shortening pauses between clauses”."
        + "\n\nSerious subject matter is not automatically slow, and whispering is not automatically slow. When explicit speed evidence is absent, use the character's stable habits, sentence rhythm, and emotion to make a natural restrained choice. Return one direction for every target segment. This stage returns directions only: do not return characters, voiceDesign, speakerId, emotion, pace, audioPlan, or any other field. Return only JSON: {\"directions\":[{\"segmentIndex\":0,\"direction\":\"fine-grained speed, rhythm, pause, breath, emphasis, and emotional direction\"}]}。"
    ),
}

_ANALYSIS_STAGE_ORDER = (
    "characters",
    "voice_design",
    "speakers",
    "delivery",
    "voice_direction",
)

_SPEAKER_PROMPTS = {
    "zh": (
        "你是有声书的说话人归属分析器。你只负责判断每个文本片段的说话人。角色名册已经由上一阶段确定，不能新建角色，不能修改角色资料，不能判断情绪和语速，不能规划背景音或音效。"
        + "\n\n"
        + _LEGACY_SPEAKER_GUIDANCE["zh"]
        + "\n\n补充执行要求：逐段返回所有 segmentIndex，不能遗漏、重复或越界；speakerId 必须是 narrator、unknown 或角色名册中的精确 id。对本章这种匿名茶肆对话，必须优先使用语义承接，不得机械套用 A-B-A-B。\n\n"
        + "关键回归样例：\n“这满街的白幡是做什么?”（茶客甲）\n“嗬，官老爷都系白腰带?”（仍是茶客甲，对同一现象作补充观察）\n“你是几日没出门了，连这都不知道?护国长公主薨了啊!举国齐丧呢!”（茶客乙，回答并解释）\n“护国长公主?你是说丹阳公主?她死了不是好事吗?该敲锣打鼓庆贺才是啊。”（茶客甲，评论回应）\n“嘘……这话被官差听见，可要抓你坐牢的。”（茶客乙，警告）\n正确分配为“甲甲乙甲乙”，不能输出机械交替的“甲乙甲乙甲”。\n\n只返回 JSON：{\"segmentAnnotations\":[{\"segmentIndex\":0,\"speakerId\":\"narrator\",\"confidence\":0.0,\"warnings\":[]}] }。不要返回 characters、emotion、pace、audioPlan 或其它字段。"
    ),
    "en": (
        "You are the speaker-attribution stage of an audiobook pipeline. Assign only the speaker for every supplied segment. The character roster is already fixed: do not create or modify characters, and do not judge emotion, pace, music, or sound effects."
        + "\n\n"
        + _LEGACY_SPEAKER_GUIDANCE["en"]
        + "\n\nAdditional stage requirements: return every segmentIndex exactly once; speakerId must be narrator, unknown, or an exact roster ID. For anonymous dialogue, prioritize semantic continuity and explicit answers, rebuttals, changes of address, and action evidence over line-number alternation. Return only speaker annotations; do not return characters, emotion, pace, audioPlan, or any other field.\n\nReturn only JSON: {\"segmentAnnotations\":[{\"segmentIndex\":0,\"speakerId\":\"narrator\",\"confidence\":0.0,\"warnings\":[]}]}."
    ),
}

_DELIVERY_PROMPTS = {
    "zh": (
        "你是有声书的表达标注分析器。角色和每个片段的说话人已经确定。你只负责为每个片段判断 emotion 和 pace，不得修改 speakerId，不得创建角色，不得生成声音提示词、场景导演、背景音乐或音效。"
        + "\n\n"
        + _LEGACY_DELIVERY_GUIDANCE["zh"]
        + "\n\n本阶段专属边界：严格保留输入中的 speakerId、角色身份和片段顺序，只补充 emotion 与 pace。必须为每个 segmentIndex 返回一次；不要返回 characters、voiceDesign、speakerId、direction、audioPlan 或其它字段。只返回 JSON：{\"segmentAnnotations\":[{\"segmentIndex\":0,\"emotion\":\"neutral\",\"pace\":\"normal\"}]}。"
    ),
    "en": (
        "You are the delivery-annotation stage of an audiobook pipeline. Characters and speakers are already fixed. Assign only emotion and pace for every segment. Do not change speakerId, create characters, write voice prompts, direct scenes, or plan music and sound effects."
        + "\n\n"
        + _LEGACY_DELIVERY_GUIDANCE["en"]
        + "\n\nStage boundary: preserve every input speakerId, character identity, and segment order; add only emotion and pace. Return every segmentIndex exactly once. Do not return characters, voiceDesign, speakerId, direction, audioPlan, or any other field. Return only JSON: {\"segmentAnnotations\":[{\"segmentIndex\":0,\"emotion\":\"neutral\",\"pace\":\"normal\"}]}."
    ),
}

_LEGACY_AUDIO_CONTEXT_GUIDANCE = {
    "zh": """从旧版整章分析规则继承以下场景和证据判断，不要只看单个片段：
1. 通读完整 chapterText、全部 segments 和 transcript，先标出场景边界、时间跳转、回忆段、动作描写、引号范围、称呼和问答关系，再规划音频。
2. 连续对白要按语义承接和场景连续性理解；不能因为片段数量多或对白一问一答，就把环境氛围清空，也不能机械地每段切换一次音乐。
3. 场景切换、时间跳转、回忆开始或结束后重新判断环境；同一场景中的连续观察、补充说明和连续叙述应保持一致的环境底色。
4. 引号中的名字、称谓、书名、诗句、公告、回忆或转述不是现场声音事件；只有正文或动作明确表示声音实际发生时，才规划对应 SFX。拟声词也要结合动作和上下文确认，不能只因出现一个声音词就凭空制造音效。
5. OCR 可能造成引号、标点和换行异常；应依据完整语义和邻近动作恢复事件边界。环境音必须服务于文本，不要把人物对白、旁白或被提及的声音当成 SFX。
6. emotion 和 pace 只是氛围与密度的参考，必须保留输入中的角色和说话人，不得在本阶段重新分配角色、改写情绪语速或生成 TTS 指导。""",
    "en": """Carry over these scene and evidence rules from the legacy chapter analyzer; do not inspect isolated lines only:
1. Read the complete chapterText, all segments, and the transcript. Mark scene boundaries, time jumps, flashbacks, actions, quotation boundaries, forms of address, and question-answer relationships before planning audio.
2. Interpret consecutive dialogue through semantic continuity and scene continuity. Do not remove the ambience merely because a scene has many dialogue segments, and do not mechanically switch music for every line.
3. Re-evaluate the environment after a scene change, time jump, or flashback begins or ends. Consecutive observations, supplements, and narration in one scene should keep a coherent environmental bed.
4. Quoted names, titles, books, poems, announcements, memories, and reported speech are not live sound events. Plan SFX only when the source or its actions clearly establish that a sound occurs. Validate onomatopoeia against context instead of inventing an effect from one sound word alone.
5. OCR may damage quotation marks, punctuation, and line breaks; use full semantics and nearby actions to recover event boundaries. Environmental audio must serve the text and must not treat dialogue, narration, or a mentioned sound as an SFX.
6. Emotion and pace are references for atmosphere and density only. Preserve the supplied characters and speakers; do not reassign speakers, rewrite emotion or pace, or generate TTS directions in this stage.""",
}


_AUDIO_PLANNER_PROMPTS = {
    "zh": (
        "你是有声书后期音频规划器和音乐与音效总监。原章节配音已经生成，并已由 Whisper 转录出带时间戳的人声。你只负责规划同主题、分层次的背景音乐和有文本证据的音效，不能修改角色、说话人、情绪、语速或 TTS 提示词。"
        "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["zh"]
        + "\n\n"
        + "输入包括：原始章节、角色和片段标注、实际人声转录及时间戳。以 transcript 时间轴作为实际听感位置，以原文、片段文本、speakerId、emotion 和 pace 补充场景与动作证据。Whisper 时间轴主要用于定位人声，不代表它已经识别出环境声；环境声必须回到原文和动作描写中核实。"
        + "\n\n"
        + "【场景覆盖、段落与音乐呼吸】只要 segments 非空，audioPlan.scenes 就不能为空，且场景必须按顺序、无重叠地连续覆盖从 segmentIndex 0 到最后一个片段。场景边界不只看地点，还要看时间、空间、人物关系、叙事功能和情绪底色；同一地点发生明显的情绪或叙事功能变化时也要拆成新的 scene。单个 scene 默认不要超过约 45 个原始片段或 60–90 秒；超过时必须拆成连续的多个 scene，不能用一个 palette 覆盖整章。每个 scene 至少要有一个 music cue，但允许在自然段落、场景转换、重要对白前后规划 2–6 秒的 musicBreak，形成听觉呼吸；短于约 20 秒的 scene 不安排 musicBreak，每个 scene 默认最多一个 break，musicBreak 不得切断一句话、问答关系、动作事件或情绪峰值。单个 scene 的 musicBreak 总时长默认不超过该 scene 实际时长的 15%，不能让整段 scene 变成无音乐；没有 break 的 cue 范围必须连续覆盖 scene。"
        + "\n\n"
        + "【主题锚点与三种编曲层次】每个 scene 默认生成 low、medium、high 三种同主题音乐变体，并在 musicPalette 中写出 eraStyle、identity、tonalCenter、mode、motif、instrumentation、register、texture、tempo、rhythmIdentity、arc 和英文 promptAnchor。promptAnchor 必须是一句可直接交给 Stable Audio 的英文主题锚点，包含时代/类型、核心乐器、调性色彩、音域、纹理和节奏身份；三个变体的 prompt 必须逐字复用同一个 promptAnchor，不能各自改写主题。low 只用于普通对白、解释和低能量叙述，采用稀疏持续音、少量音符或极轻的律动；medium 用于情绪发展、场景推进和转场，加入可辨识但克制的动机、层次和轻微脉冲；high 只用于动作、冲突或明确的情绪峰值，允许受控的低音脉冲、轻打击乐或更明显的和声张力。low/medium/high 表示编曲密度、律动和张力，不表示混音音量，禁止用 loud、quiet、volume 等词代替层次变化。三种变体必须像同一首配乐的不同编曲，而不是三个不同曲风；同一变体默认不要连续使用超过约 60 秒，除非文本明确要求稳定环境氛围。"
        + "\n\n"
        + "每个音乐变体的 prompt 必须是英文，并且以逐字一致的 promptAnchor 开头，随后只写本变体的编曲差异；必须包含 TrackType: Music, VocalType: Instrumental, no vocals, no lyrics, seamless loopable bed, no abrupt ending。音乐内容要有可辨识的纹理和动机，但由后期混音控制实际音量；不要写 low volume、barely audible、almost silent、silence、loud 或 volume。对白密集时降低旋律密度和频段拥挤度，但不能把音乐写成不可闻的噪声。low 可以禁止 heavy drums、dominant melody、strong pulse；medium 可以使用 light pulse、soft ostinato；high 可以使用 controlled pulse、restrained percussion、bass movement，但不能使用 heavy drums、sudden hits、harsh transients 或遮挡对白的频段。不要在 music prompt 中写 fire crackle、rain、crow、footsteps、door、pot boiling 等可独立录制的拟音；这些只能放入有文本证据的 SFX。音乐对象的 model 必须是 \"sm-music\"。negativePrompt 至少排除 speech、spoken words、voices、vocals、lyrics、sound effects、abrupt hits 和不相关噪声，但 high 的 negativePrompt 不得排除 percussion、pulse、bass movement 或 rhythm 本身。"
        + "\n\n"
        + "【音乐调度与情绪曲线】先根据全部文本和 transcript 的时间轴写出 scene 的 energyArc（起始状态、发展、峰值、回落/收束），再安排 musicCues；每个 cue 的 reasonZh 必须引用具体的 segmentIndex 范围、场景功能、emotion/pace、动作强度或对白密度，不能只写‘情绪推进’。musicCues 必须按顺序、不重叠、无意外缺口地引用本场景已有的 variantId；只有 musicBreak 可以造成留白。优先在 emotion、pace、动作强度或场景功能发生明确变化处切换，保持至少一个完整语义段落，不要按每句对白机械切换，也不要仅为满足变化而切换。相邻 cue 的变化应是渐进的；同一 scene 内不得反复 low→high→low 制造跳变。musicBreaks 只能出现在安全的语义边界，且每个 scene 默认最多一个。music.durationSeconds 是 Stable Audio 生成的可循环源片段长度，不是整章时长。"
        + "\n\n"
        + "音效只在原文明确或上下文强证据支持时规划：雨、风、火、鸟鸣、脚步、敲门、开门、物体碰撞、倒水、翻页等。区分持续环境音和一次性动作音；持续声可覆盖对应场景，一次性动作要短而准确。全章所有 scene 共用同一个资产 ID 命名空间，scene id、music variant id、SFX id 都必须全章唯一；SFX 不得在每个 scene 里重新从 sfx_1 编号，必须使用带 scene 前缀的稳定 ID，例如 scene_1_sfx_001。每个 SFX 必须包含 id、model、anchorText、anchorSegmentIndex、timing、durationSeconds、prompt；prompt 必须是英文并包含 TrackType: SFX。SFX 的 anchorText 必须是原文中可找到的实际文字，anchorSegmentIndex 必须有效，timing 只能是 before/during/after。SFX negativePrompt 排除 speech、spoken words、voices、music、melody、vocals、lyrics 和不相关声源。音效数量少而有用；没有音效证据时 sfx=[]，但仍必须保留 music。"
        + "\n\n"
        + "每个 scene 的 energyArc、reasonZh、musicPalette.reasonZh、每个变体 reasonZh 和每个 cue reasonZh 都必须说明文本、场景功能、emotion/pace 证据及变化原因。只返回 JSON：{\"audioPlan\":{\"version\":2,\"scenes\":[{\"id\":\"scene_1\",\"startSegmentIndex\":0,\"endSegmentIndex\":20,\"summaryZh\":\"\",\"energyArc\":\"\",\"musicPalette\":{\"eraStyle\":\"\",\"identity\":\"\",\"tonalCenter\":\"\",\"mode\":\"\",\"motif\":\"\",\"instrumentation\":\"\",\"register\":\"\",\"texture\":\"\",\"tempo\":\"\",\"rhythmIdentity\":\"\",\"promptAnchor\":\"\",\"arc\":\"\",\"reasonZh\":\"\"},\"musicVariants\":[{\"id\":\"scene_1_low\",\"level\":\"low\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"},{\"id\":\"scene_1_medium\",\"level\":\"medium\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"},{\"id\":\"scene_1_high\",\"level\":\"high\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}],\"musicCues\":[{\"id\":\"cue_1\",\"startSegmentIndex\":0,\"endSegmentIndex\":6,\"variantId\":\"scene_1_low\",\"reasonZh\":\"\"}],\"musicBreaks\":[{\"afterSegmentIndex\":6,\"durationSeconds\":4,\"reasonZh\":\"\"}],\"sfx\":[{\"id\":\"scene_1_sfx_001\",\"model\":\"sm-sfx\",\"anchorSegmentIndex\":0,\"anchorText\":\"\",\"timing\":\"during\",\"eventZh\":\"\",\"durationSeconds\":5,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}]}]}}。只有 segments 为空时才返回空 scenes。"
    ),
    "en": (
        "You are the audiobook's post-production music and sound-effects director. The chapter voice track has already been synthesized and transcribed by Whisper with timestamps. Plan same-theme, layered background music and only evidence-based sound effects. Do not modify characters, speaker attribution, emotion, pace, or TTS voice prompts."
        "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["en"]
        + "\n\n"
        + "Inputs include the source chapter, character and segment annotations, and the actual voice transcript with timestamps. Use the transcript timeline for listening position and the source text, segment text, speakerId, emotion, and pace for scene and action evidence. Whisper primarily locates speech; it does not establish environmental sounds. Verify environmental cues against the source text and actions."
        + "\n\n"
        + "[Scene coverage, sectioning, and musical breathing] When segments is non-empty, audioPlan.scenes must not be empty and must cover every segment from index 0 through the final segment in ordered, non-overlapping scene ranges. Scene boundaries are based on location, time, relationships, narrative function, and emotional bed, not location alone. A scene should normally stay within about 45 source segments or 60–90 seconds; split a longer scene into consecutive scenes instead of using one palette for the whole chapter. Every scene must have at least one music cue, but may include one 2–6 second musicBreak at a natural paragraph boundary, scene transition, or before/after important dialogue. Do not add a break to a scene shorter than about 20 seconds. Never cut a sentence, question-answer exchange, action event, or emotional peak. Total breaks should normally stay below 15% of a scene's actual duration; a scene must not become music-free. Without an intentional break, cue ranges must cover the scene contiguously."
        + "\n\n"
        + "[Theme anchor and three arrangement layers] For every scene, normally create low, medium, and high variants. musicPalette must contain eraStyle, identity, tonalCenter, mode, motif, instrumentation, register, texture, tempo, rhythmIdentity, arc, and an English promptAnchor. promptAnchor is one complete Stable Audio-ready sentence containing the period/genre, core instruments, tonal colour, register, texture, and rhythmic identity. Copy the exact same promptAnchor into every variant prompt; do not rewrite the theme per variant. Low is for ordinary dialogue, explanations, and low-energy narration: sparse sustained tones, few notes, or a very light pulse. Medium is for emotional development, movement, and transitions: a restrained motif, more layers, and a soft pulse. High is only for action, conflict, or explicit emotional peaks: controlled low-frequency motion, light percussion, or increased harmonic tension. low/medium/high describe arrangement density, rhythm, and tension, not mix volume; never use loud, quiet, or volume as the difference. The three variants must sound like different orchestrations of the same score, not unrelated genres. Do not use the same variant continuously for more than about 60 seconds unless the text clearly requires a stable ambience."
        + "\n\n"
        + "Every music variant prompt must be English, begin with the exact promptAnchor, and then state only the arrangement difference. It must contain TrackType: Music, VocalType: Instrumental, no vocals, no lyrics, seamless loopable bed, and no abrupt ending. Use identifiable texture and motif; downstream mixing controls actual loudness. Never write low volume, barely audible, almost silent, silence, loud, or volume. In dialogue-heavy sections reduce density and masking frequencies without making the bed inaudible. Low may exclude heavy drums, dominant melody, and strong pulse; medium may use light pulse and soft ostinato; high may use controlled pulse, restrained percussion, bass movement, and rhythmic motion, but must not exclude percussion, pulse, bass movement, or rhythm from its own negativePrompt. Avoid heavy drums, sudden hits, harsh transients, and frequencies that mask speech. Do not put fire crackle, rain, crow calls, footsteps, doors, boiling pots, or other diegetic SFX in a music prompt; plan them separately only with textual evidence. Adjacent variants must preserve the exact promptAnchor and change gradually. Music objects must use model \"sm-music\". Common music negativePrompt terms should exclude speech, spoken words, voices, vocals, lyrics, sound effects, abrupt hits, and unrelated noise, but must not exclude music itself."
        + "\n\n"
        + "[Music scheduling and emotional arc] First infer each scene's energyArc (opening state, development, peak, and release/closure) from the full text and transcript timeline, then schedule musicCues. Every cue reasonZh must cite a concrete segment range and the scene function, emotion/pace, action intensity, or dialogue density; do not write only \"emotional development\". musicCues must be ordered, non-overlapping, and cover the scene without accidental gaps while referencing only variant IDs declared in the same scene; only musicBreak may create silence. Switch variants at meaningful changes in emotion, pace, action intensity, or scene function, preserve at least one complete semantic paragraph, and never switch mechanically for every dialogue line or merely to create variation. Adjacent cue changes must be gradual; do not repeatedly jump low→high→low inside one scene. musicBreaks may occur only at safe semantic boundaries and normally no more than one per scene. music.durationSeconds is the loopable Stable Audio source duration, not the full chapter duration."
        + "\n\n"
        + "Plan SFX only when the source explicitly contains or strongly supports an audible event such as fire, rain, wind, birds, footsteps, knocking, doors, object impacts, pouring, or page turns. Distinguish sustained ambience from short one-shot actions. All scenes share one chapter-wide asset ID namespace: scene IDs, music variant IDs, and SFX IDs must be unique across the entire chapter. Never restart SFX numbering at sfx_1 in each scene; use a stable scene-prefixed ID such as scene_1_sfx_001. Every SFX must include id, model, anchorText, anchorSegmentIndex, timing, durationSeconds, and prompt; the prompt must be English and contain TrackType: SFX. anchorText must be exact source text, anchorSegmentIndex must be in range, and timing must be before, during, or after. SFX negativePrompt should exclude speech, spoken words, voices, music, melody, vocals, lyrics, and unrelated sources. Keep SFX minimal and useful; when no SFX evidence exists, return sfx=[], but still return music."
        + "\n\n"
        + "For every scene, energyArc, musicPalette, each variant, each cue, and reasonZh must explain the source text, scene function, emotion/pace evidence, and why it changes. Return only JSON: {\"audioPlan\":{\"version\":2,\"scenes\":[{\"id\":\"scene_1\",\"startSegmentIndex\":0,\"endSegmentIndex\":20,\"summaryZh\":\"\",\"energyArc\":\"\",\"musicPalette\":{\"eraStyle\":\"\",\"identity\":\"\",\"tonalCenter\":\"\",\"mode\":\"\",\"motif\":\"\",\"instrumentation\":\"\",\"register\":\"\",\"texture\":\"\",\"tempo\":\"\",\"rhythmIdentity\":\"\",\"promptAnchor\":\"\",\"arc\":\"\",\"reasonZh\":\"\"},\"musicVariants\":[{\"id\":\"scene_1_low\",\"level\":\"low\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"},{\"id\":\"scene_1_medium\",\"level\":\"medium\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"},{\"id\":\"scene_1_high\",\"level\":\"high\",\"model\":\"sm-music\",\"durationSeconds\":30,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}],\"musicCues\":[{\"id\":\"cue_1\",\"startSegmentIndex\":0,\"endSegmentIndex\":6,\"variantId\":\"scene_1_low\",\"reasonZh\":\"\"}],\"musicBreaks\":[{\"afterSegmentIndex\":6,\"durationSeconds\":4,\"reasonZh\":\"\"}],\"sfx\":[{\"id\":\"scene_1_sfx_001\",\"model\":\"sm-sfx\",\"anchorSegmentIndex\":0,\"anchorText\":\"\",\"timing\":\"during\",\"eventZh\":\"\",\"durationSeconds\":5,\"prompt\":\"\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}]}]}}. Return an empty scenes array only when segments is empty."
    ),
}

# The legacy identity, speaker, and delivery rules are injected into their
# corresponding split stages above.  Dynamic directing remains isolated in
# the private voice_direction stage, and the planner keeps only audio fields.
_DELIVERY_PROMPTS["zh"] = _DELIVERY_PROMPTS["zh"].replace(
    "不得生成声音提示词、场景导演、背景音乐或音效",
    "不得生成声音提示词、背景音乐或音效",
)


# Audio planning is deliberately split into narrow contracts.  The old
# all-in-one prompt above is kept as a compatibility reference for existing
# fixtures, but real requests use the three prompts below so a model never
# has to reason about scene boundaries, music assets, and SFX in one response.
AUDIO_PLANNING_STAGE_ORDER = ("scene_structure", "music", "sfx")
_AUDIO_PLANNING_BATCH_MAX_CHARACTERS = 9_000

_AUDIO_SCENE_STRUCTURE_PROMPTS = {
    "zh": (
        "你是有声书后期制作的场景结构分析器。原章节配音和 Whisper 转录已经完成。你只负责把本章按听觉连续性划分为连续场景，并概括每个场景的能量曲线；不要生成背景音乐、音效、Stable Audio prompt 或任何 TTS 提示词。"
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["zh"]
        + "\n\n"
        + "必须通读完整 chapterText 和全部 segments。场景边界依据地点、时间、空间、人物关系、叙事功能、动作密度和情绪底色；同一地点发生明显的叙事功能或能量变化时也要拆分。连续对白、补充叙述和同一动作过程应保持在同一场景内，不得按每句对白切换。每个场景通常不超过约 45 个片段；场景必须按顺序、无重叠地连续覆盖全部片段。"
        + "\n\n只返回 JSON：{\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"summaryZh\":\"场景摘要\",\"energyArc\":\"低能量观察→逐步推进→克制收束\"}]}。不得返回 music、musicPalette、musicVariants、musicCues、musicBreaks、sfx 或其它字段。"
    ),
    "en": (
        "You are the scene-structure stage of an audiobook post-production pipeline. The chapter voice track and Whisper transcript are already complete. Divide the chapter into coherent listening scenes and describe each scene's energy arc. Do not generate music, SFX, Stable Audio prompts, or TTS directions."
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["en"]
        + "\n\n"
        + "Read the complete chapterText and all segments. Boundaries depend on location, time, space, relationships, narrative function, action density, and emotional bed. Keep consecutive dialogue, supplementary narration, and one continuous action in the same scene; never switch scenes for every dialogue line. A scene should normally stay under about 45 segments. Scenes must be ordered, non-overlapping, and cover every segment."
        + "\n\nReturn only JSON: {\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"summaryZh\":\"scene summary\",\"energyArc\":\"low observation -> gradual movement -> restrained closure\"}]}. Do not return music, musicPalette, musicVariants, musicCues, musicBreaks, sfx, or any other field."
    ),
}

_AUDIO_MUSIC_PROMPTS = {
    "zh": (
        "你是有声书后期的背景音乐设计器。场景结构已经由上一阶段确定；你只为输入的场景设计同主题、分层次的背景音乐资产和音乐调度，不负责重新划分场景，也不生成音效。"
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["zh"]
        + "\n\n"
        + "每个输入 scene 必须原样使用其 id 和 segment 范围，只返回对应的 musicPalette、musicVariants、musicCues、musicBreaks。每个 scene 默认生成 low、medium、high 三个同主题变体。三种变体必须逐字复用同一个短英文 promptAnchor；low 稀疏低能量，medium 有克制动机和轻脉冲，high 只在明确动作、冲突或情绪峰值使用受控脉冲、轻打击乐和更明显的和声张力。low/medium/high 是编曲密度和张力，不是音量。musicPalette 可以保留详细分析，但不要把整段 palette 复制进 Stable Audio prompt。"
        + "\n\n"
        + "官方 Stable Audio 音乐提示词优先使用 Genre、Instruments、Mood/energy 和 BPM。promptAnchor 必须是英文单行，约不超过 180 个英文字符，格式类似：TrackType: Music, VocalType: Instrumental, Genre: historical ambient, Instruments: guqin, xiao, low strings, 68 BPM, restrained suspense, sparse texture。只保留一个类型、1–3 种核心乐器、情绪/能量和速度，不要写完整场景叙事、镜头画面、混音音量或同义限制。每个完整 prompt 尽量不超过约 260 个英文字符。三种变体只在末尾追加很短的编曲差异，例如 sparse sustained texture、restrained motif and light pulse、controlled pulse and light percussion。不要重复写 no vocals、no lyrics、seamless loopable bed、no abrupt ending；VocalType: Instrumental 已足够。禁止把雨、风、火、脚步、敲门、开门等拟音写入音乐 prompt；这些交给 SFX 阶段。negativePrompt 简短排除 speech、vocals、lyrics、sound effects、abrupt hits，但 high 不得排除 percussion、pulse、bass movement 或 rhythm。"
        + "\n\n"
        + "musicCues 必须按顺序、不重叠地覆盖该 scene 的全部片段，只有 musicBreak 可以在安全语义边界形成 2–6 秒留白。切换应依据 emotion、pace、动作强度、对白密度或叙事功能变化，不能按每句对白机械切换。每个 cue 的 reasonZh 必须引用具体片段范围和变化证据。只返回 JSON：{\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"musicPalette\":{\"eraStyle\":\"\",\"identity\":\"\",\"tonalCenter\":\"\",\"mode\":\"\",\"motif\":\"\",\"instrumentation\":\"\",\"register\":\"\",\"texture\":\"\",\"tempo\":\"\",\"rhythmIdentity\":\"\",\"promptAnchor\":\"\",\"arc\":\"\",\"reasonZh\":\"\"},\"musicVariants\":[],\"musicCues\":[],\"musicBreaks\":[]}]}。不得返回 sfx 或重新修改场景范围。"
    ),
    "en": (
        "You are the background-music design stage of an audiobook post-production pipeline. Scene structure is already fixed. For the supplied scenes, design same-theme layered music assets and music scheduling only; do not redraw scenes and do not generate SFX."
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["en"]
        + "\n\n"
        + "For every supplied scene, preserve its exact id and segment range and return only musicPalette, musicVariants, musicCues, and musicBreaks. Normally create low, medium, and high variants of one theme. Copy one identical short English promptAnchor into all three prompts. Follow Stable Audio's compact music structure: TrackType, VocalType, Genre, Instruments, mood/energy, and optional BPM. Keep the anchor to about 180 characters and the complete prompt to about 260 characters; do not copy the full scene description or palette into it. Low is sparse, medium adds a restrained motif and soft pulse, and high adds controlled pulse or light percussion. These levels describe arrangement density and tension, not volume."
        + "\n\n"
        + "Every music prompt must be English and begin with the exact same `TrackType: Music, VocalType: Instrumental` promptAnchor. Do not repeat no vocals, no lyrics, seamless loopable bed, or no abrupt ending. Do not put rain, wind, fire, footsteps, knocking, doors, or other diegetic SFX in music prompts. Keep negativePrompt short: speech, vocals, lyrics, sound effects, abrupt hits; high must not exclude percussion, pulse, bass movement, or rhythm."
        + "\n\n"
        + "musicCues must be ordered, non-overlapping, and cover every segment in the scene; only a musicBreak may create a 2–6 second gap at a safe semantic boundary. Switch at meaningful changes in emotion, pace, action intensity, dialogue density, or narrative function, never mechanically for every dialogue line. Every cue reasonZh must cite a concrete segment range and evidence. Return only JSON: {\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"musicPalette\":{\"eraStyle\":\"\",\"identity\":\"\",\"tonalCenter\":\"\",\"mode\":\"\",\"motif\":\"\",\"instrumentation\":\"\",\"register\":\"\",\"texture\":\"\",\"tempo\":\"\",\"rhythmIdentity\":\"\",\"promptAnchor\":\"\",\"arc\":\"\",\"reasonZh\":\"\"},\"musicVariants\":[],\"musicCues\":[],\"musicBreaks\":[]}]}。Do not return sfx or change scene ranges."
    ),
}

_AUDIO_SFX_PROMPTS = {
    "zh": (
        "你是有声书后期的音效证据分析器。场景结构已经确定；你只负责从输入场景的原文、片段和局部 Whisper 转录中找出确实发生且值得生成的音效，不负责音乐、场景边界、角色、情绪或 TTS。"
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["zh"]
        + "\n\n"
        + "只有正文或动作明确支持时才生成 SFX；引号中的名字、称谓、书名、诗句、公告、回忆、转述和单独出现的拟声词不能自动变成音效。区分持续环境声和一次性动作声，音效要少而有用，不能用 SFX 替代背景音乐。每个音效必须使用 scene 前缀的全章唯一 id，anchorText 必须能在对应片段原文中找到，anchorSegmentIndex 必须属于该 scene，timing 只能是 before/during/after。prompt 使用英文并包含 TrackType: SFX，negativePrompt 排除 speech、spoken words、voices、music、melody、vocals、lyrics 和无关声源。没有证据时返回 sfx=[]。只返回 JSON：{\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"sfx\":[{\"id\":\"scene_001_sfx_001\",\"model\":\"sm-sfx\",\"anchorSegmentIndex\":0,\"anchorText\":\"原文短语\",\"timing\":\"during\",\"eventZh\":\"\",\"durationSeconds\":3,\"prompt\":\"TrackType: SFX, ...\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}]}]}。不得返回 music 或修改 scene 范围。"
    ),
    "en": (
        "You are the evidence-based SFX stage of an audiobook post-production pipeline. Scene structure is fixed. Find only audible events clearly supported by the source text, scene segments, and local Whisper transcript. Do not plan music, redraw scenes, or modify characters, emotion, pace, or TTS."
        + "\n\n"
        + _LEGACY_AUDIO_CONTEXT_GUIDANCE["en"]
        + "\n\n"
        + "Create SFX only when the source or actions clearly establish an event. Quoted names, titles, books, poems, announcements, memories, reported speech, and an isolated onomatopoeia are not automatically live sounds. Distinguish sustained ambience from short one-shot actions; keep effects minimal and never replace the music bed. Every effect needs a chapter-wide scene-prefixed unique id, exact anchorText found in the source segment, a valid anchorSegmentIndex inside the scene, and timing before/during/after. Prompts must be English and contain TrackType: SFX; negative prompts exclude speech, spoken words, voices, music, melody, vocals, lyrics, and unrelated sources. Return sfx=[] when there is no evidence. Return only JSON: {\"scenes\":[{\"id\":\"scene_001\",\"startSegmentIndex\":0,\"endSegmentIndex\":12,\"sfx\":[{\"id\":\"scene_001_sfx_001\",\"model\":\"sm-sfx\",\"anchorSegmentIndex\":0,\"anchorText\":\"exact source phrase\",\"timing\":\"during\",\"eventZh\":\"\",\"durationSeconds\":3,\"prompt\":\"TrackType: SFX, ...\",\"negativePrompt\":\"\",\"reasonZh\":\"\"}]}]}。Do not return music or change scene ranges."
    ),
}


def _language_key(language: str) -> str:
    return "zh" if str(language).lower().startswith("zh") else "en"


def _segment_payload(request: ChapterAnalysisRequest) -> list[dict[str, Any]]:
    return [
        {
            "segmentIndex": index,
            "type": segment.type,
            "text": segment.text,
            "startOffset": segment.start_offset,
            "endOffset": segment.end_offset,
            "speakerHint": segment.speaker_hint,
            "warnings": segment.warnings,
        }
        for index, segment in enumerate(
            segment_dialogue(request.text, language=request.language)
        )
    ]


def _known_payload(
    characters: list[CharacterContext | CharacterAnalysis],
    *,
    include_voice_design: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for character in characters:
        if isinstance(character, CharacterContext):
            item = {
                "id": character.id,
                "canonicalName": character.canonical_name,
                "aliases": character.aliases,
                "gender": character.gender,
                "ageClass": character.age_class,
            }
            if include_voice_design and character.voice_design:
                item["voiceDesign"] = character.voice_design
            result.append(item)
        else:
            item = {
                "id": character.id,
                "canonicalName": character.canonical_name,
                "aliases": character.aliases,
                "gender": character.gender,
                "ageClass": character.age_class,
            }
            if include_voice_design and character.voice_design:
                item["voiceDesign"] = character.voice_design
            result.append(item)
    return result


def _parse_characters(payload: dict[str, Any]) -> list[CharacterAnalysis]:
    characters: list[CharacterAnalysis] = []
    for item in payload.get("characters", []):
        if not isinstance(item, dict) or not item.get("id") or not item.get("canonicalName"):
            continue
        aliases = item.get("aliases", [])
        characters.append(
            CharacterAnalysis(
                id=str(item["id"]),
                canonical_name=str(item["canonicalName"]),
                aliases=[str(alias) for alias in aliases] if isinstance(aliases, list) else [],
                gender=str(item.get("gender", "unknown")),
                age_class=str(item.get("ageClass", "unknown")),
                confidence=float(item.get("confidence", 0.0)),
                voice_design=str(item.get("voiceDesign", "") or "").strip(),
            )
        )
    return characters


def _parse_voice_designs(payload: dict[str, Any]) -> dict[str, str]:
    designs: dict[str, str] = {}
    for item in payload.get("characters", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        design = str(item.get("voiceDesign", "") or "").strip()
        if design:
            designs[str(item["id"])] = design
    return designs


def _parse_voice_directions(payload: dict[str, Any]) -> dict[int, str]:
    directions: dict[int, str] = {}
    raw_items = payload.get("directions")
    if not isinstance(raw_items, list) or not raw_items:
        raw_items = payload.get("segmentAnnotations", [])
    for item in raw_items:
        if not isinstance(item, dict) or "segmentIndex" not in item:
            continue
        direction = str(
            item.get("direction")
            or item.get("voiceDirection")
            or item.get("performanceDirection")
            or ""
        ).strip()
        if not direction:
            emotion = _normalize_emotion(item.get("emotion", "neutral"))
            pace = _normalize_pace(item.get("pace", "normal"))
            direction = f"保持{pace}语速，呈现{emotion}的自然演绎。"
        if direction:
            directions[int(item["segmentIndex"])] = direction
    return directions


def _parse_speakers(payload: dict[str, Any]) -> list[SegmentAnnotation]:
    annotations: list[SegmentAnnotation] = []
    for item in payload.get("segmentAnnotations", []):
        if not isinstance(item, dict) or "segmentIndex" not in item:
            continue
        warnings = item.get("warnings", [])
        annotations.append(
            SegmentAnnotation(
                segment_index=int(item["segmentIndex"]),
                speaker_id=str(item.get("speakerId", "unknown")),
                emotion="neutral",
                confidence=float(item.get("confidence", 0.0)),
                warnings=[str(warning) for warning in warnings] if isinstance(warnings, list) else [],
                pace="normal",
            )
        )
    return annotations


def _parse_delivery(payload: dict[str, Any]) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for item in payload.get("segmentAnnotations", []):
        if not isinstance(item, dict) or "segmentIndex" not in item:
            continue
        result[int(item["segmentIndex"])] = (
            _normalize_emotion(item.get("emotion", "neutral")),
            _normalize_pace(item.get("pace", "normal")),
        )
    return result


def _publish(request: ChapterAnalysisRequest, stage: str, value: dict[str, Any]) -> None:
    if request.stage_callback is not None:
        request.stage_callback(stage, value)


def _should_reuse_cached_stage(request: ChapterAnalysisRequest, stage: str) -> bool:
    """Return whether a persisted stage may be used for this retry.

    A cache is deliberately opt-in.  This prevents a normal re-analysis from
    silently reusing an older model response, while allowing a failed run to
    resume from the first unfinished stage.  ``script`` means all analysis
    stages completed and only script assembly needs to be retried.
    """

    from_stage = request.resume_from_stage
    if not from_stage:
        return False
    if from_stage == "script":
        return True
    try:
        return _ANALYSIS_STAGE_ORDER.index(stage) < _ANALYSIS_STAGE_ORDER.index(from_stage)
    except ValueError as error:
        raise ValueError(
            "resume_from_stage must be characters, voice_design, speakers, delivery, voice_direction, or script"
        ) from error


def _cached_stage_payload(
    request: ChapterAnalysisRequest,
    stage: str,
    required_key: str,
) -> dict[str, Any] | None:
    if not _should_reuse_cached_stage(request, stage):
        return None
    payload = request.cached_stages.get(stage)
    if not isinstance(payload, dict) or not isinstance(payload.get(required_key), list):
        return None
    return payload


def _direction_segment_payload(
    segments: list[dict[str, Any]],
    annotations: list[SegmentAnnotation],
) -> list[dict[str, Any]]:
    annotation_by_index = {item.segment_index: item for item in annotations}
    result: list[dict[str, Any]] = []
    for segment in segments:
        index = int(segment["segmentIndex"])
        annotation = annotation_by_index.get(index)
        result.append(
            {
                "segmentIndex": index,
                "type": segment.get("type"),
                "text": segment.get("text", ""),
                "speakerId": annotation.speaker_id if annotation else "unknown",
                "emotion": annotation.emotion if annotation else "neutral",
                "pace": annotation.pace if annotation else "normal",
            }
        )
    return result


def _direction_batches(
    segments: list[dict[str, Any]],
    annotations: list[SegmentAnnotation],
    *,
    max_characters: int = 18000,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Batch dynamic directions by payload size while retaining local context."""
    prepared = _direction_segment_payload(segments, annotations)
    if not prepared:
        return []
    batches: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    start = 0
    while start < len(prepared):
        end = start
        size = 0
        while end < len(prepared):
            candidate_size = len(str(prepared[end].get("text", ""))) + 180
            if end > start and size + candidate_size > max_characters:
                break
            size += candidate_size
            end += 1
        target = prepared[start:end]
        context = prepared[max(0, start - 2):min(len(prepared), end + 2)]
        batches.append((target, context))
        start = end
    return batches


def run_split_chapter_analysis(analyzer: Any, request: ChapterAnalysisRequest) -> ChapterAnalysisResult:
    """Run the independent identity, voice, attribution, delivery, and direction calls."""
    if request.resume_from_stage not in {None, *_ANALYSIS_STAGE_ORDER, "script"}:
        raise ValueError(
            "resume_from_stage must be characters, voice_design, speakers, delivery, voice_direction, or script"
        )
    request = replace(request, language=resolve_text_language(request.text, request.language))
    segments = _segment_payload(request)
    base = {
        "chapterId": request.chapter_id,
        "language": request.language,
        "chapterText": request.text,
        "segments": segments,
        "knownCharacters": _known_payload(request.known_characters),
    }

    character_payload = dict(base)
    cached_characters = _cached_stage_payload(request, "characters", "characters")
    if cached_characters is None:
        character_payload_result = analyzer._request_stage_json(
            _CHARACTER_PROMPTS[_language_key(request.language)],
            character_payload,
        )
    else:
        character_payload_result = cached_characters
    characters = _parse_characters(character_payload_result)
    _publish(
        request,
        "characters",
        {
            "characters": [
                {
                    "id": item.id,
                    "canonicalName": item.canonical_name,
                    "aliases": item.aliases,
                    "gender": item.gender,
                    "ageClass": item.age_class,
                    "confidence": item.confidence,
                }
                for item in characters
            ]
        },
    )
    roster = [*request.known_characters, *characters]

    # This is deliberately a separate LLM call from character identity
    # analysis.  Existing character/speaker/delivery prompts and response
    # shapes remain unchanged.
    voice_design_payload = dict(base)
    voice_design_payload["characters"] = _known_payload(
        characters,
        include_voice_design=True,
    )
    voice_design_payload["knownCharacters"] = _known_payload(
        request.known_characters,
        include_voice_design=True,
    )
    cached_voice_design = _cached_stage_payload(
        request, "voice_design", "characters"
    )
    if cached_voice_design is None:
        if characters:
            voice_design_payload_result = analyzer._request_stage_json(
                _VOICE_DESIGN_PROMPTS[_language_key(request.language)],
                voice_design_payload,
            )
        else:
            voice_design_payload_result = {"characters": []}
    else:
        voice_design_payload_result = cached_voice_design
    generated_designs = _parse_voice_designs(voice_design_payload_result)
    known_designs = {
        item.id: item.voice_design
        for item in request.known_characters
        if item.voice_design
    }
    characters = [
        replace(
            item,
            voice_design=generated_designs.get(item.id)
            or item.voice_design
            or known_designs.get(item.id, ""),
        )
        for item in characters
    ]
    _publish(
        request,
        "voice_design",
        {
            "characters": [
                {
                    "id": item.id,
                    "canonicalName": item.canonical_name,
                    "voiceDesign": item.voice_design,
                }
                for item in characters
                if item.voice_design
            ]
        },
    )
    roster = [*request.known_characters, *characters]

    speaker_payload = dict(base)
    speaker_payload["knownCharacters"] = _known_payload(roster)
    cached_speakers = _cached_stage_payload(
        request, "speakers", "segmentAnnotations"
    )
    if cached_speakers is None:
        speaker_payload_result = analyzer._request_stage_json(
            _SPEAKER_PROMPTS[_language_key(request.language)],
            speaker_payload,
        )
    else:
        speaker_payload_result = cached_speakers
    speaker_annotations = _parse_speakers(speaker_payload_result)
    expected_count = len(segments)
    if sorted(item.segment_index for item in speaker_annotations) != list(range(expected_count)):
        raise ValueError(
            "invalid segment annotations in speaker stage: expected every segment index exactly once"
        )
    _publish(
        request,
        "speakers",
        {
            "segmentAnnotations": [
                {
                    "segmentIndex": item.segment_index,
                    "speakerId": item.speaker_id,
                    "confidence": item.confidence,
                    "warnings": item.warnings,
                }
                for item in speaker_annotations
            ]
        },
    )

    delivery_payload = dict(speaker_payload)
    delivery_payload["speakerAnnotations"] = [
        {
            "segmentIndex": item.segment_index,
            "speakerId": item.speaker_id,
            "confidence": item.confidence,
            "warnings": item.warnings,
        }
        for item in speaker_annotations
    ]
    cached_delivery = _cached_stage_payload(
        request, "delivery", "segmentAnnotations"
    )
    if cached_delivery is None:
        delivery_payload_result = analyzer._request_stage_json(
            _DELIVERY_PROMPTS[_language_key(request.language)],
            delivery_payload,
        )
    else:
        delivery_payload_result = cached_delivery
    delivery = _parse_delivery(delivery_payload_result)
    if sorted(delivery) != list(range(expected_count)):
        raise ValueError(
            "invalid segment annotations in delivery stage: expected every segment index exactly once"
        )
    _publish(
        request,
        "delivery",
        {
            "segmentAnnotations": [
                {
                    "segmentIndex": index,
                    "emotion": emotion,
                    "pace": pace,
                }
                for index, (emotion, pace) in sorted(delivery.items())
            ]
        },
    )

    annotations = [
        replace(item, emotion=delivery[item.segment_index][0], pace=delivery[item.segment_index][1])
        for item in sorted(speaker_annotations, key=lambda value: value.segment_index)
    ]

    # Dynamic directions are generated in a separate batched LLM phase.  The
    # stage is cached as an internal artifact and is intentionally not attached
    # to the public chapter script or analysis result shown in the UI.
    direction_batches = _direction_batches(segments, annotations)
    cached_voice_direction = _cached_stage_payload(
        request, "voice_direction", "directions"
    )
    if cached_voice_direction is not None:
        voice_directions = _parse_voice_directions(cached_voice_direction)
    else:
        voice_directions: dict[int, str] = {}
        character_payload = _known_payload(characters, include_voice_design=True)
        for target_segments, context_segments in direction_batches:
            direction_payload = {
                "chapterId": request.chapter_id,
                "language": request.language,
                "characters": character_payload,
                "targetSegments": target_segments,
                "contextSegments": context_segments,
            }
            batch_result = analyzer._request_stage_json(
                _VOICE_DIRECTION_PROMPTS[_language_key(request.language)],
                direction_payload,
            )
            voice_directions.update(_parse_voice_directions(batch_result))

    if sorted(voice_directions) != list(range(expected_count)):
        missing = [index for index in range(expected_count) if index not in voice_directions]
        raise ValueError(
            "invalid directions in voice_direction stage: expected every segment index exactly once; "
            f"missing {missing[:10]}"
        )
    _publish(
        request,
        "voice_direction",
        {
            "directions": [
                {"segmentIndex": index, "direction": voice_directions[index]}
                for index in sorted(voice_directions)
            ]
        },
    )

    return _normalize_and_validate_analysis(
        ChapterAnalysisResult(
            characters=characters,
            segment_annotations=annotations,
            audio_plan=ChapterAudioPlan(),
            voice_directions=voice_directions,
        ),
        request=request,
        segment_count=expected_count,
        segment_texts=[str(segment["text"]) for segment in segments],
    )


def _audio_stage_value(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("audioPlan", payload)
    return value if isinstance(value, dict) else {}


def _audio_raw_scenes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_scenes = _audio_stage_value(payload).get("scenes", [])
    return [item for item in raw_scenes if isinstance(item, dict)] if isinstance(raw_scenes, list) else []


def _audio_segment_payload(segment: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "segmentIndex": int(segment.get("segmentIndex", fallback_index)),
        "type": str(segment.get("type") or "narration"),
        "text": str(segment.get("text") or ""),
        "speakerId": str(segment.get("speakerId") or "narrator"),
        "emotion": str(segment.get("emotion") or "neutral"),
        "pace": str(segment.get("pace") or "normal"),
    }
    return item


def _audio_segments_for_range(
    request: AudioPlanningRequest,
    start_segment_index: int,
    end_segment_index: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fallback_index, segment in enumerate(request.segments):
        if not isinstance(segment, dict):
            continue
        index = int(segment.get("segmentIndex", fallback_index))
        if start_segment_index <= index <= end_segment_index:
            result.append(_audio_segment_payload(segment, fallback_index))
    return sorted(result, key=lambda item: int(item["segmentIndex"]))


def _compact_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in transcript:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        compact: dict[str, Any] = {"text": text}
        for key in ("start", "end"):
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                compact[key] = float(value)
        result.append(compact)
    return result


def _transcript_for_range(
    request: AudioPlanningRequest,
    start_segment_index: int,
    end_segment_index: int,
) -> list[dict[str, Any]]:
    """Keep only a local transcript window for a scene.

    Whisper timestamps do not carry source-segment IDs.  A proportional slice
    with two neighboring items on each side is a safe compact approximation for
    planning; the source segment text and indices remain the authoritative cue
    anchors.
    """

    transcript = _compact_transcript(request.transcript)
    if not transcript or not request.segments:
        return transcript
    segment_count = max(1, len(request.segments))
    start = max(0, min(segment_count - 1, start_segment_index))
    end = max(start, min(segment_count - 1, end_segment_index))
    left = max(0, math.floor(len(transcript) * start / segment_count) - 2)
    right = min(
        len(transcript),
        max(left + 1, math.ceil(len(transcript) * (end + 1) / segment_count)) + 2,
    )
    return transcript[left:right]


def _audio_scene_context(
    request: AudioPlanningRequest,
    scene: AudioScenePlan,
) -> dict[str, Any]:
    scene_segments = _audio_segments_for_range(
        request,
        scene.start_segment_index,
        scene.end_segment_index,
    )
    return {
        "id": scene.id,
        "startSegmentIndex": scene.start_segment_index,
        "endSegmentIndex": scene.end_segment_index,
        "summaryZh": scene.summary_zh,
        "energyArc": scene.energy_arc,
        "sceneText": "\n".join(
            f"[{item['segmentIndex']}] {item['text']}" for item in scene_segments
        ),
        "segments": scene_segments,
        "transcript": _transcript_for_range(
            request,
            scene.start_segment_index,
            scene.end_segment_index,
        ),
    }


def _audio_scene_batches(
    request: AudioPlanningRequest,
    scenes: list[AudioScenePlan],
) -> list[list[AudioScenePlan]]:
    batches: list[list[AudioScenePlan]] = []
    current: list[AudioScenePlan] = []
    current_size = 0
    for scene in scenes:
        scene_size = len(
            json.dumps(
                _audio_scene_context(request, scene),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        if current and current_size + scene_size > _AUDIO_PLANNING_BATCH_MAX_CHARACTERS:
            batches.append(current)
            current = []
            current_size = 0
        current.append(scene)
        current_size += scene_size
    if current:
        batches.append(current)
    return batches


def _scene_structure_dict(scene: AudioScenePlan) -> dict[str, Any]:
    return {
        "id": scene.id,
        "startSegmentIndex": scene.start_segment_index,
        "endSegmentIndex": scene.end_segment_index,
        "summaryZh": scene.summary_zh,
        "energyArc": scene.energy_arc,
    }


def _audio_scene_stage_dict(
    scene: AudioScenePlan,
    *,
    include_music: bool,
    include_sfx: bool,
) -> dict[str, Any]:
    item = _scene_structure_dict(scene)
    if include_music:
        if scene.music is not None:
            item["music"] = {
                "model": scene.music.model,
                "durationSeconds": scene.music.duration_seconds,
                "prompt": scene.music.prompt,
                "negativePrompt": scene.music.negative_prompt,
                "reasonZh": scene.music.reason_zh,
            }
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
                "afterSegmentIndex": music_break.after_segment_index,
                "durationSeconds": music_break.duration_seconds,
                "reasonZh": music_break.reason_zh,
            }
            for music_break in scene.music_breaks
        ]
    if include_sfx:
        item["sfx"] = [
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
        ]
    return item


def _audio_stage_payload(
    scenes: list[AudioScenePlan],
    *,
    include_music: bool,
    include_sfx: bool,
) -> dict[str, Any]:
    return {
        "version": 2,
        "scenes": [
            _audio_scene_stage_dict(
                scene,
                include_music=include_music,
                include_sfx=include_sfx,
            )
            for scene in scenes
        ],
    }


def _normalize_scene_structure(
    scenes: list[AudioScenePlan],
    segment_count: int,
) -> list[AudioScenePlan]:
    if segment_count <= 0:
        return []

    candidates = sorted(
        scenes,
        key=lambda scene: (scene.start_segment_index, scene.end_segment_index),
    )
    normalized: list[AudioScenePlan] = []
    used_ids: set[str] = set()
    cursor = 0
    gap_number = 1

    def unique_id(value: str, fallback: str) -> str:
        base = value.strip() or fallback
        candidate = base
        suffix = 2
        while candidate in used_ids:
            candidate = f"{base}_{suffix:03d}"
            suffix += 1
        used_ids.add(candidate)
        return candidate

    def append_gap(start: int, end: int) -> None:
        nonlocal gap_number
        if start > end:
            return
        normalized.append(
            AudioScenePlan(
                id=unique_id(f"scene_gap_{gap_number:03d}", f"scene_gap_{gap_number:03d}"),
                start_segment_index=start,
                end_segment_index=end,
                summary_zh="补齐连续场景范围",
                energy_arc="保持当前叙事底色",
            )
        )
        gap_number += 1

    for index, scene in enumerate(candidates, start=1):
        start = max(cursor, scene.start_segment_index, 0)
        end = min(segment_count - 1, scene.end_segment_index)
        if end < start:
            continue
        if start > cursor:
            append_gap(cursor, start - 1)
        normalized.append(
            replace(
                scene,
                id=unique_id(scene.id, f"scene_{index:03d}"),
                start_segment_index=start,
                end_segment_index=end,
                music=None,
                music_palette={},
                music_variants=[],
                music_cues=[],
                music_breaks=[],
                sfx=[],
            )
        )
        cursor = end + 1

    if cursor < segment_count:
        append_gap(cursor, segment_count - 1)
    if not normalized:
        append_gap(0, segment_count - 1)
    return normalized


def _split_long_audio_scenes(
    request: AudioPlanningRequest,
    scenes: list[AudioScenePlan],
) -> list[AudioScenePlan]:
    """Bound scene requests even when the structure model returns one huge scene."""

    if not scenes:
        return scenes
    segment_count = len(request.segments)
    if segment_count <= 0:
        return scenes

    text_lengths = [
        len(str(segment.get("text") or ""))
        if isinstance(segment, dict)
        else 0
        for segment in request.segments
    ]
    total_text_length = max(1, sum(text_lengths))
    transcript_end = max(
        (
            float(item.get("end"))
            for item in request.transcript
            if isinstance(item, dict)
            and isinstance(item.get("end"), (int, float))
            and math.isfinite(float(item.get("end")))
        ),
        default=0.0,
    )
    max_scene_seconds = 90.0
    max_scene_segments = 45
    expanded: list[AudioScenePlan] = []

    for scene in scenes:
        start = max(0, scene.start_segment_index)
        end = min(segment_count - 1, scene.end_segment_index)
        length = end - start + 1
        text_before = sum(text_lengths[:start])
        text_through = sum(text_lengths[: end + 1])
        estimated_seconds = transcript_end * max(
            0.0,
            (text_through - text_before) / total_text_length,
        )
        part_count = max(
            1,
            math.ceil(length / max_scene_segments),
            math.ceil(estimated_seconds / max_scene_seconds),
        )
        if part_count == 1:
            expanded.append(scene)
            continue

        for part in range(part_count):
            part_start = start + (length * part) // part_count
            part_end = start + (length * (part + 1)) // part_count - 1
            expanded.append(
                replace(
                    scene,
                    id=f"{scene.id}_part_{part + 1:03d}",
                    start_segment_index=part_start,
                    end_segment_index=part_end,
                    summary_zh=(
                        f"{scene.summary_zh}（听觉分段 {part + 1}/{part_count}）"
                    ).strip("（）"),
                )
            )
    return expanded


def _audio_should_reuse_cached_stage(
    request: AudioPlanningRequest,
    stage: str,
) -> bool:
    from_stage = request.resume_from_stage
    if not from_stage:
        return False
    if from_stage == "complete":
        return True
    try:
        return AUDIO_PLANNING_STAGE_ORDER.index(stage) < AUDIO_PLANNING_STAGE_ORDER.index(from_stage)
    except ValueError as error:
        raise ValueError(
            "resume_from_stage must be scene_structure, music, sfx, or complete"
        ) from error


def _cached_audio_stage(
    request: AudioPlanningRequest,
    stage: str,
) -> dict[str, Any] | None:
    if not _audio_should_reuse_cached_stage(request, stage):
        return None
    payload = request.cached_stages.get(stage)
    if not isinstance(payload, dict) or not isinstance(_audio_stage_value(payload).get("scenes"), list):
        return None
    return payload


def _publish_audio_stage(
    request: AudioPlanningRequest,
    stage: str,
    payload: dict[str, Any],
) -> None:
    if request.stage_callback is not None:
        request.stage_callback(stage, payload)


def _parse_stage_scenes(payload: dict[str, Any]) -> list[AudioScenePlan]:
    return _parse_audio_plan(_audio_stage_value(payload)).scenes


def _merge_audio_stage_scenes(
    base_scenes: list[AudioScenePlan],
    stage_scenes: list[AudioScenePlan],
    *,
    stage: str,
) -> list[AudioScenePlan]:
    by_id = {scene.id: scene for scene in stage_scenes}
    merged: list[AudioScenePlan] = []
    for base in base_scenes:
        addon = by_id.get(base.id)
        if addon is None:
            merged.append(base)
            continue
        if stage == "music":
            merged.append(
                replace(
                    base,
                    music=addon.music,
                    music_palette=addon.music_palette,
                    music_variants=addon.music_variants,
                    music_cues=addon.music_cues,
                    music_breaks=addon.music_breaks,
                )
            )
        else:
            merged.append(replace(base, sfx=addon.sfx))
    return merged


def _run_audio_asset_stage(
    analyzer: Any,
    request: AudioPlanningRequest,
    scenes: list[AudioScenePlan],
    *,
    stage: str,
    prompt: str,
) -> list[AudioScenePlan]:
    """Run music or SFX over bounded scene batches, never over the full chapter."""

    additions: list[AudioScenePlan] = []
    for batch in _audio_scene_batches(request, scenes):
        expected = {scene.id: scene for scene in batch}
        result = analyzer._request_stage_json(
            prompt,
            {
                "chapterId": request.chapter_id,
                "language": request.language,
                "scenes": [_audio_scene_context(request, scene) for scene in batch],
            },
            stage_name=f"audio_{stage}",
        )
        raw_scenes = _audio_raw_scenes(result)
        prepared: list[dict[str, Any]] = []
        for raw_index, raw_scene in enumerate(raw_scenes):
            item = dict(raw_scene)
            scene_id = str(item.get("id") or "").strip()
            target = expected.get(scene_id)
            if target is None and len(raw_scenes) == len(batch) and raw_index < len(batch):
                target = batch[raw_index]
            if target is None:
                continue
            item["id"] = target.id
            item.setdefault("startSegmentIndex", target.start_segment_index)
            item.setdefault("endSegmentIndex", target.end_segment_index)
            prepared.append(item)
        additions.extend(_parse_stage_scenes({"version": 2, "scenes": prepared}))

    return _merge_audio_stage_scenes(scenes, additions, stage=stage)


def run_audio_planning(analyzer: Any, request: AudioPlanningRequest) -> ChapterAudioPlan:
    """Run scene, music, and SFX planning as bounded, resumable LLM stages."""

    request = replace(request, language=resolve_text_language(request.text, request.language))
    if request.resume_from_stage not in {None, *AUDIO_PLANNING_STAGE_ORDER, "complete"}:
        raise ValueError(
            "resume_from_stage must be scene_structure, music, sfx, or complete"
        )
    segment_count = len(request.segments)
    if segment_count <= 0:
        return ChapterAudioPlan()

    cached_structure = _cached_audio_stage(request, "scene_structure")
    if cached_structure is None:
        structure_result = analyzer._request_stage_json(
            _AUDIO_SCENE_STRUCTURE_PROMPTS[_language_key(request.language)],
            {
                "chapterId": request.chapter_id,
                "language": request.language,
                "chapterText": request.text,
                "segments": [
                    _audio_segment_payload(segment, index)
                    for index, segment in enumerate(request.segments)
                    if isinstance(segment, dict)
                ],
                "characters": select_active_audio_characters(
                    request.characters,
                    request.segments,
                ),
            },
            stage_name="audio_scene_structure",
        )
        structure_scenes = _split_long_audio_scenes(
            request,
            _normalize_scene_structure(
                _parse_stage_scenes(structure_result),
                segment_count,
            ),
        )
    else:
        structure_scenes = _split_long_audio_scenes(
            request,
            _normalize_scene_structure(
                _parse_stage_scenes(cached_structure),
                segment_count,
            ),
        )
    structure_payload = {
        "version": 2,
        "scenes": [_scene_structure_dict(scene) for scene in structure_scenes],
    }
    _publish_audio_stage(request, "scene_structure", structure_payload)

    cached_music = _cached_audio_stage(request, "music")
    if cached_music is None:
        music_scenes = _run_audio_asset_stage(
            analyzer,
            request,
            structure_scenes,
            stage="music",
            prompt=_AUDIO_MUSIC_PROMPTS[_language_key(request.language)],
        )
    else:
        music_scenes = _merge_audio_stage_scenes(
            structure_scenes,
            _parse_stage_scenes(cached_music),
            stage="music",
        )
    music_payload = _audio_stage_payload(
        music_scenes,
        include_music=True,
        include_sfx=False,
    )
    _publish_audio_stage(request, "music", music_payload)

    cached_sfx = _cached_audio_stage(request, "sfx")
    if cached_sfx is None:
        sfx_scenes = _run_audio_asset_stage(
            analyzer,
            request,
            music_scenes,
            stage="sfx",
            prompt=_AUDIO_SFX_PROMPTS[_language_key(request.language)],
        )
    else:
        sfx_scenes = _merge_audio_stage_scenes(
            music_scenes,
            _parse_stage_scenes(cached_sfx),
            stage="sfx",
        )
    sfx_payload = _audio_stage_payload(
        sfx_scenes,
        include_music=True,
        include_sfx=True,
    )
    _publish_audio_stage(request, "sfx", sfx_payload)

    validated_plan = _validate_audio_plan(
        ChapterAudioPlan(scenes=sfx_scenes, version=2),
        segment_count=segment_count,
        segment_texts=[str(segment.get("text", "")) for segment in request.segments],
    )
    return ensure_audio_music_coverage(
        validated_plan,
        segment_count=segment_count,
        language=request.language,
        segments=request.segments,
    )
