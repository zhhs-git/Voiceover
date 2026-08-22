import json
from dataclasses import replace

import pytest

from audiobook_worker import llm as llm_module
from audiobook_worker.llm import (
    AudioScenePlan,
    AudioPlanningRequest,
    ChapterAudioPlan,
    CharacterContext,
    ChapterAnalysisRequest,
    MockLLMAnalyzer,
    MusicPlan,
    OpenAICompatibleAnalyzer,
    OpenAICompatibleConfig,
    SfxPlan,
    audio_plan_to_dict,
    analyzer_from_models_config,
    ensure_audio_music_coverage,
    normalize_audio_plan_anchors,
    resolve_model,
    resolve_model_from_config,
    select_active_audio_characters,
)
from audiobook_worker.llm_env import LlmEnvironment


def test_mock_adapter_returns_characters_speaker_emotion_and_confidence():
    analyzer = MockLLMAnalyzer()
    request = ChapterAnalysisRequest(
        book_id="book_123",
        chapter_id="chapter_001",
        text='"Come in," Elizabeth said. Darcy waited.',
        language="en",
    )

    result = analyzer.analyze_chapter(request)

    assert result.characters[0].canonical_name == "Elizabeth"
    assert result.characters[0].gender == "female"
    assert result.segment_annotations[0].speaker_id == "elizabeth"
    assert result.segment_annotations[0].emotion == "neutral"
    assert result.segment_annotations[0].confidence >= 0.7


def test_select_active_audio_characters_excludes_inactive_roster_and_narrator():
    selected = select_active_audio_characters(
        [
            {
                "id": "active",
                "canonicalName": "本章角色",
                "aliases": ["阿甲"],
                "gender": "male",
                "ageClass": "adult",
                "voiceId": "mimo_active",
                "voiceDesign": "低沉、克制，吐字清晰。",
                "unrelatedMetadata": "must not be sent",
            },
            {
                "id": "inactive",
                "canonicalName": "其他章节角色",
                "gender": "female",
                "ageClass": "adult",
                "voiceId": "mimo_inactive",
            },
            {
                "id": "narrator",
                "canonicalName": "旁白",
                "gender": "female",
                "ageClass": "adult",
            },
        ],
        [
            {"segmentIndex": 0, "speakerId": "active"},
            {"segmentIndex": 1, "speakerId": "narrator"},
        ],
    )

    assert selected == [
        {
            "id": "active",
            "canonicalName": "本章角色",
            "aliases": ["阿甲"],
            "gender": "male",
            "ageClass": "adult",
            "voiceId": "mimo_active",
            "voiceDesign": "低沉、克制，吐字清晰。",
        }
    ]


def test_audio_planning_payload_only_contains_active_characters():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {"choices": [{"message": {"content": json.dumps({"audioPlan": {"scenes": []}})}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    analyzer.plan_audio(
        AudioPlanningRequest(
            book_id="book_zh",
            chapter_id="chapter_001",
            text="他推开门。",
            language="zh",
            segments=[
                {"segmentIndex": 0, "text": "他推开门。", "speakerId": "active"},
                {"segmentIndex": 1, "text": "门开了。", "speakerId": "narrator"},
            ],
            transcript=[{"start": 0.0, "end": 1.0, "text": "他推开门。"}],
            characters=[
                {
                    "id": "active",
                    "canonicalName": "本章角色",
                    "gender": "male",
                    "ageClass": "adult",
                    "voiceDesign": "低沉、克制。",
                },
                {
                    "id": "inactive",
                    "canonicalName": "全书其他角色",
                    "gender": "female",
                    "ageClass": "adult",
                    "voiceDesign": "明亮。",
                },
            ],
        )
    )

    planner_input = json.loads(calls[0]["messages"][1]["content"])
    assert [character["id"] for character in planner_input["characters"]] == ["active"]
    assert planner_input["characters"][0]["voiceDesign"] == "低沉、克制。"


def test_audio_planning_splits_requests_and_resumes_after_sfx_failure():
    calls = []
    saved = {}
    scene_response = {
        "scenes": [{
            "id": "scene_001",
            "startSegmentIndex": 0,
            "endSegmentIndex": 1,
            "summaryZh": "雨夜",
            "energyArc": "观察→推进",
        }]
    }
    music_response = {
        "scenes": [{
            "id": "scene_001",
            "musicVariants": [{
                "id": "scene_001_low",
                "level": "low",
                "model": "sm-music",
                "durationSeconds": 30,
                "prompt": "TrackType: Music, VocalType: Instrumental, no vocals, no lyrics, seamless loopable bed, no abrupt ending, sparse",
                "negativePrompt": "speech, vocals",
                "reasonZh": "普通叙述铺底",
            }],
            "musicCues": [{
                "id": "scene_001_cue_001",
                "startSegmentIndex": 0,
                "endSegmentIndex": 1,
                "variantId": "scene_001_low",
                "reasonZh": "连续覆盖本场景",
            }],
            "musicBreaks": [],
        }]
    }
    sfx_response = {"scenes": [{"id": "scene_001", "sfx": []}]}

    def response(value):
        return {"choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)}}]}

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        if len(calls) == 1:
            return response(scene_response)
        if len(calls) == 2:
            return response(music_response)
        raise RuntimeError("simulated SFX outage")

    def save_stage(stage, payload):
        saved[stage] = payload

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="openai",
            api_key="test-key",
            base_url="http://test/v1",
            model="test-model",
            max_retries=1,
        ),
        transport=transport,
    )
    request = AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="雨落在街上。有人停下脚步。",
        language="zh",
        segments=[
            {"segmentIndex": 0, "text": "雨落在街上。"},
            {"segmentIndex": 1, "text": "有人停下脚步。"},
        ],
        transcript=[{"start": 0.0, "end": 1.0, "text": "雨落在街上。"}],
        stage_callback=save_stage,
    )

    with pytest.raises(RuntimeError, match="audio_sfx"):
        analyzer.plan_audio(request)

    assert set(saved) == {"scene_structure", "music"}
    assert "transcript" not in json.loads(calls[0]["messages"][1]["content"])
    assert len(calls) == 3

    resume_calls = []

    def resume_transport(url, headers, payload, timeout_seconds):
        resume_calls.append(payload)
        return response(sfx_response)

    resumed_analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="openai",
            api_key="test-key",
            base_url="http://test/v1",
            model="test-model",
            max_retries=1,
        ),
        transport=resume_transport,
    )
    resumed = resumed_analyzer.plan_audio(
        replace(request, stage_callback=None, cached_stages=saved, resume_from_stage="sfx")
    )

    assert len(resume_calls) == 1
    assert "音效证据分析器" in resume_calls[0]["messages"][0]["content"]
    assert resumed.scenes[0].music_variants[0].id == "scene_001_low"


def test_normalizes_sfx_anchor_across_segments_and_chapter_start():
    plan = ChapterAudioPlan(
        scenes=[
            AudioScenePlan(
                id="scene_001",
                start_segment_index=0,
                end_segment_index=2,
                summary_zh="木门开启",
                sfx=[
                    SfxPlan(
                        id="sfx_001",
                        model="sm-sfx",
                        anchor_segment_index=0,
                        timing="before",
                        event_zh="木门吱呀声",
                        duration_seconds=2,
                        prompt="TrackType: SFX, wooden door creak",
                        negative_prompt="music, speech",
                        reason_zh="正文明确出现门声",
                        anchor_text="门“吱呀”一声被推开",
                    )
                ],
            )
        ]
    )

    normalized = normalize_audio_plan_anchors(
        plan,
        ["门", "吱呀", "一声被推开"],
    )

    effect = normalized.scenes[0].sfx[0]
    assert effect.anchor_segment_index == 0
    assert effect.timing == "during"


def test_normalizes_sfx_anchor_to_segment_containing_exact_cue():
    plan = ChapterAudioPlan(
        scenes=[
            AudioScenePlan(
                id="scene_001",
                start_segment_index=0,
                end_segment_index=1,
                summary_zh="木门开启",
                sfx=[
                    SfxPlan(
                        id="sfx_001",
                        model="sm-sfx",
                        anchor_segment_index=0,
                        timing="during",
                        event_zh="吱呀声",
                        duration_seconds=2,
                        prompt="TrackType: SFX, wooden door creak",
                        negative_prompt="music, speech",
                        reason_zh="正文明确出现门声",
                        anchor_text="吱呀",
                    )
                ],
            )
        ]
    )

    normalized = normalize_audio_plan_anchors(plan, ["门", "吱呀"])

    assert normalized.scenes[0].sfx[0].anchor_segment_index == 1


def test_serialized_audio_plan_namespaces_duplicate_sfx_ids_by_scene():
    effect = SfxPlan(
        id="sfx_1",
        model="sm-sfx",
        anchor_segment_index=0,
        timing="during",
        event_zh="雨声",
        duration_seconds=2,
        prompt="TrackType: SFX, rain",
        negative_prompt="music, speech",
        reason_zh="正文明确出现雨声",
    )
    serialized = audio_plan_to_dict(
        ChapterAudioPlan(
            scenes=[
                AudioScenePlan(
                    id="scene_1",
                    start_segment_index=0,
                    end_segment_index=0,
                    summary_zh="开场",
                    sfx=[effect],
                ),
                AudioScenePlan(
                    id="scene_2",
                    start_segment_index=1,
                    end_segment_index=1,
                    summary_zh="转场",
                    sfx=[effect],
                ),
            ]
        )
    )

    assert [
        scene["sfx"][0]["id"] for scene in serialized["scenes"]
    ] == ["sfx_1", "scene_2_sfx_1"]


def test_real_backend_configuration_is_declared_but_disabled_by_default():
    analyzer = MockLLMAnalyzer()

    assert analyzer.backend_id == "mock"
    assert analyzer.supports_real_model is False


def test_resolves_default_deepseek_model_from_pi_models_config():
    resolved = resolve_model_from_config(
        {
            "default": "deepseek/deepseek-v4-pro",
            "providers": {
                "deepseek": {
                    "baseUrl": "https://api.deepseek.com/v1",
                    "api": "openai-completions",
                    "apiKey": "test-key",
                    "family": "deepseek",
                    "models": [
                        {
                            "id": "deepseek/deepseek-v4-pro",
                            "name": "DeepSeek V4 Pro",
                            "maxTokens": 384000,
                        }
                    ],
                }
            },
        }
    )

    assert resolved.provider == "deepseek"
    assert resolved.model_id == "deepseek/deepseek-v4-pro"
    assert resolved.base_url == "https://api.deepseek.com/v1"
    assert resolved.api_key == "test-key"
    assert resolved.max_tokens == 384000
    assert resolved.supports_response_format is False


def test_project_llm_environment_overrides_legacy_provider_url_and_key(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "read_models_json",
        lambda: {
            "default": "legacy/legacy-model",
            "providers": {
                "legacy": {
                    "baseUrl": "https://legacy.example/v1",
                    "apiKey": "legacy-secret",
                    "family": "legacy",
                    "models": [{"id": "legacy-model", "maxTokens": 4096}],
                }
            },
        },
    )
    monkeypatch.setattr(
        llm_module,
        "read_llm_environment",
        lambda: LlmEnvironment(
            model_id="provider/project-model",
            base_url="https://project.example/v1",
            api_key="project-secret",
        ),
    )

    resolved = resolve_model()

    assert resolved is not None
    assert resolved.model_id == "provider/project-model"
    assert resolved.base_url == "https://project.example/v1"
    assert resolved.api_key == "project-secret"
    assert resolved.provider == "env"


def test_project_llm_environment_does_not_depend_on_a_valid_legacy_catalog(monkeypatch):
    def broken_catalog():
        raise json.JSONDecodeError("bad catalog", "{", 1)

    monkeypatch.setattr(
        llm_module,
        "read_models_json",
        broken_catalog,
    )
    monkeypatch.setattr(
        llm_module,
        "read_llm_environment",
        lambda: LlmEnvironment(
            model_id="provider/project-model",
            base_url="https://project.example/v1",
            api_key="project-secret",
        ),
    )

    resolved = resolve_model()

    assert resolved is not None
    assert resolved.model_id == "provider/project-model"
    assert resolved.base_url == "https://project.example/v1"
    assert resolved.api_key == "project-secret"


def test_can_switch_between_deepseek_pro_and_flash_models_from_config():
    config = {
        "default": "deepseek/deepseek-v4-pro",
        "providers": {
            "deepseek": {
                "baseUrl": "https://api.deepseek.com/v1",
                "api": "openai-completions",
                "apiKey": "test-key",
                "family": "deepseek",
                "models": [
                    {"id": "deepseek/deepseek-v4-pro", "maxTokens": 384000},
                    {"id": "deepseek/deepseek-v4-flash", "maxTokens": 384000},
                ],
            }
        },
    }

    pro = analyzer_from_models_config(config, "deepseek/deepseek-v4-pro")
    flash = analyzer_from_models_config(config, "deepseek/deepseek-v4-flash")

    assert isinstance(pro, OpenAICompatibleAnalyzer)
    assert isinstance(flash, OpenAICompatibleAnalyzer)
    assert pro.config.model == "deepseek/deepseek-v4-pro"
    assert flash.config.model == "deepseek/deepseek-v4-flash"


def test_openai_compatible_adapter_posts_chat_completion_request_from_resolved_model():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "characters": [
                                    {
                                        "id": "elizabeth",
                                        "canonicalName": "Elizabeth",
                                        "aliases": ["Lizzy"],
                                        "gender": "female",
                                        "ageClass": "adult",
                                        "confidence": 0.94,
                                    }
                                ],
                                "segmentAnnotations": [
                                    {
                                        "segmentIndex": 0,
                                        "speakerId": "elizabeth",
                                        "emotion": "happy",
                                        "pace": "fast",
                                        "confidence": 0.88,
                                        "warnings": [],
                                    },
                                    {
                                        "segmentIndex": 1,
                                        "speakerId": "narrator",
                                        "emotion": "neutral",
                                        "pace": "fast",
                                        "confidence": 0.99,
                                        "warnings": [],
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek/deepseek-v4-pro",
            max_tokens=384000,
            supports_response_format=True,
        ),
        transport=transport,
    )

    result = analyzer.analyze_chapter(
        ChapterAnalysisRequest(
            book_id="book_123",
            chapter_id="chapter_001",
            text='"Hello," Elizabeth said.',
            language="en",
        )
    )

    assert analyzer.backend_id == "deepseek"
    assert analyzer.supports_real_model is True
    assert calls[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["payload"]["model"] == "deepseek/deepseek-v4-pro"
    assert calls[0]["payload"]["max_tokens"] == 384000
    assert calls[0]["payload"]["response_format"] == {"type": "json_object"}
    assert calls[0]["payload"]["thinking"] == {"type": "disabled"}
    assert result.characters[0].canonical_name == "Elizabeth"
    assert result.segment_annotations[0].emotion == "happy"
    assert result.segment_annotations[0].pace == "fast"
    assert result.segment_annotations[1].pace == "fast"
    assert result.audio_plan.scenes == []


def test_json_response_format_is_omitted_by_default_for_compatibility_gateways():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="openai",
            api_key="test-key",
            base_url="http://gateway.test/v1",
            model="gpt-5.6-luna",
            max_retries=1,
        ),
        transport=transport,
    )

    result = analyzer._request_stage_json(
        "Return only JSON.",
        {"probe": True},
        stage_name="compatibility_probe",
    )

    assert result == {"ok": True}
    assert "response_format" not in calls[0]


def test_json_response_format_can_be_enabled_from_model_config():
    analyzer = analyzer_from_models_config(
        {
            "default": "openai/gpt-5.6-luna",
            "providers": {
                "openai": {
                    "baseUrl": "http://gateway.test/v1",
                    "api": "openai-completions",
                    "apiKey": "test-key",
                    "models": [
                        {
                            "id": "gpt-5.6-luna",
                            "supportsResponseFormat": True,
                        }
                    ],
                }
            },
        }
    )

    assert isinstance(analyzer, OpenAICompatibleAnalyzer)
    assert analyzer.config.supports_response_format is True


def test_audio_planning_parses_stable_audio_scene_and_sfx_prompts():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": json.dumps({
                "characters": [],
                "segmentAnnotations": [{
                    "segmentIndex": 0,
                    "speakerId": "narrator",
                    "emotion": "neutral",
                    "pace": "normal",
                    "confidence": 0.99,
                    "warnings": [],
                }],
                "audioPlan": {
                    "scenes": [{
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 0,
                        "summaryZh": "雨夜街道与木门开启",
                        "music": {
                            "model": "sm-music",
                            "durationSeconds": 30,
                            "prompt": "Dark historical Chinese suspense instrumental, sparse guqin and xiao, low strings, restrained tension, atmospheric rain-soaked stone street, spacious cinematic reverb, no vocals",
                            "negativePrompt": "vocals, lyrics, speech, bright pop melody",
                            "reasonZh": "给旁白提供低存在感的悬疑氛围",
                        },
                        "sfx": [{
                            "id": "sfx_001",
                            "model": "sm-sfx",
                            "anchorSegmentIndex": 0,
                            "timing": "during",
                            "eventZh": "雨声与木门吱呀声",
                            "durationSeconds": 5,
                            "prompt": "TrackType: SFX, heavy rain falling on wet stone pavement, followed by a slow wooden door creak in a dim old courtyard, natural room ambience, realistic field recording, short decay",
                            "negativePrompt": "music, melody, speech, voices",
                            "reasonZh": "强化文本中明确出现的雨和木门声音",
                        }],
                    }],
                },
            }, ensure_ascii=False)}}]
        }

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    plan = analyzer.plan_audio(AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="雨落在青石板上。木门吱呀一声打开。",
        language="zh",
        segments=[
            {"segmentIndex": 0, "text": "雨落在青石板上。"},
            {"segmentIndex": 1, "text": "木门吱呀一声打开。"},
        ],
        transcript=[
            {"start": 0.0, "end": 0.8, "text": "雨落在青石板上。"},
            {"start": 0.8, "end": 1.6, "text": "木门吱呀一声打开。"},
        ],
    ))

    scene = plan.scenes[0]
    assert scene.id == "scene_001"
    assert scene.music is not None
    assert scene.music.model == "sm-music"
    assert "instrumental" in scene.music.prompt
    assert scene.sfx[0].model == "sm-sfx"
    assert scene.sfx[0].anchor_segment_index == 0
    assert "TrackType: SFX" in scene.sfx[0].prompt
    assert set(audio_plan_to_dict(plan)["scenes"][0]) == {
        "id",
        "startSegmentIndex",
        "endSegmentIndex",
        "summaryZh",
        "music",
        "sfx",
    }
    assert len(calls) == 3
    assert "场景结构分析器" in calls[0]["messages"][0]["content"]
    assert "连续覆盖" in calls[0]["messages"][0]["content"]
    assert "场景边界" in calls[0]["messages"][0]["content"]
    assert "energyArc" in calls[0]["messages"][0]["content"]
    assert "背景音乐设计器" in calls[1]["messages"][0]["content"]
    assert "promptAnchor" in calls[1]["messages"][0]["content"]
    assert "high 不得排除" in calls[1]["messages"][0]["content"]
    assert "音效证据分析器" in calls[2]["messages"][0]["content"]
    assert "TrackType: SFX" in calls[2]["messages"][0]["content"]
    assert "全章唯一" in calls[2]["messages"][0]["content"]
    assert "narrationDirection" not in calls[0]["messages"][0]["content"]
    planner_input = json.loads(calls[0]["messages"][1]["content"])
    assert planner_input["characters"] == []
    music_input = json.loads(calls[1]["messages"][1]["content"])
    assert music_input["scenes"][0]["transcript"][0]["start"] == 0.0


def test_audio_planning_drops_malformed_optional_audio_shapes_without_failing():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [],
            "segmentAnnotations": [{
                "segmentIndex": 0,
                "speakerId": "narrator",
                "emotion": "neutral",
                "confidence": 0.99,
                "warnings": [],
            }],
            "audioPlan": {
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 0,
                    "summaryZh": "模型把可选音频字段写成了错误类型",
                    # This is the shape returned by the failing request.  A
                    # malformed optional asset must not fail chapter analysis.
                    "music": "a quiet instrumental bed",
                    "sfx": "没有音效",
                }],
            },
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    plan = analyzer.plan_audio(AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="院子里很安静。",
        language="zh",
        segments=[{"segmentIndex": 0, "text": "院子里很安静。"}],
        transcript=[{"start": 0.0, "end": 1.0, "text": "院子里很安静。"}],
    ))

    assert len(plan.scenes) == 1
    assert plan.scenes[0].start_segment_index == 0
    assert plan.scenes[0].end_segment_index == 0
    assert plan.scenes[0].music is not None
    assert "TrackType: Music, VocalType: Instrumental" in plan.scenes[0].music.prompt
    assert len(plan.scenes[0].music.prompt) <= 260
    assert plan.scenes[0].sfx == []


def test_audio_planning_parses_same_theme_music_variants_cues_and_breaks():
    def transport(url, headers, payload, timeout_seconds):
        return {
            "choices": [{
                "message": {
                    "content": json.dumps(
                        {
                            "audioPlan": {
                                "version": 2,
                                "scenes": [{
                                    "id": "scene_001",
                                    "startSegmentIndex": 0,
                                    "endSegmentIndex": 2,
                                    "summaryZh": "雨夜街道",
                                    "energyArc": "低能量雨夜观察→情绪推进→克制收束",
                                    "musicPalette": {
                                        "identity": "冷静克制的历史悬疑底色",
                                        "instrumentation": "低音弦乐与箫",
                                        "register": "低到中音域",
                                        "texture": "稀疏持续纹理",
                                        "tempo": "缓慢至中速",
                                        "reasonZh": "保持雨夜场景统一听感",
                                    },
                                    "musicVariants": [
                                        {
                                            "id": "scene_001_low",
                                            "level": "low",
                                            "model": "sm-music",
                                            "durationSeconds": 30,
                                            "prompt": "TrackType: Music, VocalType: Instrumental, seamless loopable bed, sparse low strings",
                                        },
                                        {
                                            "id": "scene_001_medium",
                                            "level": "medium",
                                            "model": "sm-music",
                                            "durationSeconds": 30,
                                            "prompt": "TrackType: Music, VocalType: Instrumental, seamless loopable bed, restrained pulse",
                                        },
                                        {
                                            "id": "scene_001_high",
                                            "level": "high",
                                            "model": "sm-music",
                                            "durationSeconds": 30,
                                            "prompt": "TrackType: Music, VocalType: Instrumental, seamless loopable bed, tense rising strings",
                                        },
                                    ],
                                    "musicCues": [
                                        {
                                            "id": "cue_001",
                                            "startSegmentIndex": 0,
                                            "endSegmentIndex": 1,
                                            "variantId": "scene_001_low",
                                            "reasonZh": "普通对白使用低能量铺底",
                                        },
                                        {
                                            "id": "cue_002",
                                            "startSegmentIndex": 2,
                                            "endSegmentIndex": 2,
                                            "variantId": "scene_001_medium",
                                            "reasonZh": "情绪推进时提高能量",
                                        },
                                    ],
                                    "musicBreaks": [{
                                        "afterSegmentIndex": 1,
                                        "durationSeconds": 4,
                                        "reasonZh": "场景转折前留白",
                                    }],
                                    "sfx": [],
                                }],
                            }
                        },
                        ensure_ascii=False,
                    ),
                }
            }]
        }

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    plan = analyzer.plan_audio(AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="雨落在街上。有人停下脚步。",
        language="zh",
        segments=[
            {"segmentIndex": 0, "text": "雨落在街上。"},
            {"segmentIndex": 1, "text": "有人停下脚步。"},
            {"segmentIndex": 2, "text": "他抬头看向远处。"},
        ],
        transcript=[
            {"start": 0.0, "end": 1.0, "text": "雨落在街上。"},
            {"start": 1.0, "end": 2.0, "text": "有人停下脚步。"},
            {"start": 2.0, "end": 3.0, "text": "他抬头看向远处。"},
        ],
    ))

    scene = plan.scenes[0]
    assert plan.version == 2
    assert scene.music is None
    assert scene.energy_arc == "低能量雨夜观察→情绪推进→克制收束"
    assert {variant.level for variant in scene.music_variants} == {
        "low",
        "medium",
        "high",
    }
    assert [cue.variant_id for cue in scene.music_cues] == [
        "scene_001_low",
        "scene_001_medium",
    ]
    assert scene.music_breaks[0].duration_seconds == pytest.approx(4)
    serialized = audio_plan_to_dict(plan)
    assert serialized["version"] == 2
    assert serialized["scenes"][0]["energyArc"] == "低能量雨夜观察→情绪推进→克制收束"
    assert serialized["scenes"][0]["musicVariants"][0]["id"] == "scene_001_low"


def test_audio_planning_preserves_partial_music_scenes_and_fills_missing_music():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "audioPlan": {
                "scenes": [
                    {
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 0,
                        "summaryZh": "开场",
                        "music": {
                            "model": "sm-music",
                            "durationSeconds": 30,
                            "prompt": "TrackType: Music, VocalType: Instrumental, continuous opening bed",
                        },
                        "sfx": [{
                            "id": "sfx_001",
                            "model": "sm-sfx",
                            "anchorSegmentIndex": 0,
                            "anchorText": "雨",
                            "timing": "during",
                            "durationSeconds": 3,
                            "prompt": "TrackType: SFX, rain",
                        }],
                    },
                    {
                        "id": "scene_002",
                        "startSegmentIndex": 1,
                        "endSegmentIndex": 2,
                        "summaryZh": "对白",
                        "music": None,
                        "sfx": [{
                            "id": "sfx_002",
                            "model": "sm-sfx",
                            "anchorSegmentIndex": 2,
                            "anchorText": "敲门",
                            "timing": "during",
                            "durationSeconds": 2,
                            "prompt": "TrackType: SFX, knocking",
                        }],
                    },
                ],
            },
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    plan = analyzer.plan_audio(AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="雨落下来。有人说话。敲门声响起。",
        language="zh",
        segments=[
            {"segmentIndex": 0, "text": "雨落下来。"},
            {"segmentIndex": 1, "text": "有人说话。"},
            {"segmentIndex": 2, "text": "敲门声响起。"},
        ],
        transcript=[
            {"start": 0.0, "end": 1.0, "text": "雨落下来。"},
            {"start": 1.0, "end": 2.0, "text": "有人说话。"},
            {"start": 2.0, "end": 3.0, "text": "敲门声响起。"},
        ],
    ))

    assert len(plan.scenes) == 2
    assert plan.scenes[0].start_segment_index == 0
    assert plan.scenes[0].end_segment_index == 0
    assert plan.scenes[0].music is not None
    assert plan.scenes[1].start_segment_index == 1
    assert plan.scenes[1].end_segment_index == 2
    assert plan.scenes[1].music is not None
    assert plan.scenes[1].music.prompt != plan.scenes[0].music.prompt
    assert {effect.id for scene in plan.scenes for effect in scene.sfx} == {
        "sfx_001",
        "sfx_002",
    }


def test_long_music_scene_is_split_into_emotion_aware_phases():
    segments = [
        {
            "text": "营地里的对话继续。" if index < 39 else "紧张的脚步逼近。",
            "emotion": (
                "neutral" if index < 39 else "tense" if index < 78 else "sad"
            ),
            "pace": "normal",
        }
        for index in range(117)
    ]
    base_music = MusicPlan(
        model="sm-music",
        duration_seconds=30.0,
        prompt="TrackType: Music, VocalType: Instrumental, restrained audiobook bed",
        negative_prompt="speech, vocals",
        reason_zh="持续铺底",
    )
    source_effect = SfxPlan(
        id="sfx_step",
        model="sm-sfx",
        anchor_segment_index=45,
        timing="during",
        event_zh="脚步",
        duration_seconds=3.0,
        prompt="TrackType: SFX, footsteps",
        negative_prompt="speech",
        reason_zh="文本出现脚步",
    )

    plan = ensure_audio_music_coverage(
        ChapterAudioPlan(
            scenes=[
                AudioScenePlan(
                    id="scene_1",
                    start_segment_index=0,
                    end_segment_index=116,
                    summary_zh="营地到冲突",
                    music=base_music,
                    sfx=[source_effect],
                )
            ]
        ),
        segment_count=117,
        segments=segments,
    )

    assert [(scene.start_segment_index, scene.end_segment_index) for scene in plan.scenes] == [
        (0, 38),
        (39, 77),
        (78, 116),
    ]
    assert len({scene.music.prompt for scene in plan.scenes if scene.music}) == 3
    assert [effect.id for scene in plan.scenes for effect in scene.sfx] == ["sfx_step"]
    assert plan.scenes[1].sfx[0].anchor_segment_index == 45


def test_audio_planning_normalizes_friendly_or_missing_model_labels():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "audioPlan": {
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 0,
                    "summaryZh": "营地火堆",
                    "music": {
                        "model": "music",
                        "durationSeconds": 30,
                        "prompt": "TrackType: Music, VocalType: Instrumental, sparse camp ambience",
                    },
                    "sfx": [{
                        "id": "sfx_001",
                        "anchorSegmentIndex": 0,
                        "anchorText": "火堆噼啪作响",
                        "timing": "during",
                        "durationSeconds": 5,
                        "prompt": "TrackType: SFX, realistic fire crackle",
                    }],
                }],
            },
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    plan = analyzer.plan_audio(AudioPlanningRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="火堆噼啪作响。",
        language="zh",
        segments=[{"segmentIndex": 0, "text": "火堆噼啪作响。"}],
        transcript=[{"start": 0.0, "end": 1.0, "text": "火堆噼啪作响。"}],
    ))

    assert plan.scenes[0].music is not None
    assert plan.scenes[0].music.model == "sm-music"
    assert plan.scenes[0].sfx[0].model == "sm-sfx"


def test_analysis_rejects_invalid_audio_scene_range():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [],
            "segmentAnnotations": [{
                "segmentIndex": 0,
                "speakerId": "narrator",
                "emotion": "neutral",
                "confidence": 0.99,
                "warnings": [],
            }],
            "audioPlan": {
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 1,
                    "summaryZh": "越界场景",
                    "music": None,
                    "sfx": [],
                }],
            },
        })}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    result = analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="院子里很安静。",
        language="zh",
    ))

    assert result.audio_plan.scenes == []


def test_real_adapter_overrides_stale_english_language_for_chinese_prompt():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": json.dumps({
                "characters": [{
                    "id": "张三",
                    "canonicalName": "张三",
                    "aliases": [],
                    "gender": "male",
                    "ageClass": "adult",
                    "confidence": 0.95,
                }],
                "segmentAnnotations": [
                    {
                        "segmentIndex": 0,
                        "speakerId": "narrator",
                        "emotion": "neutral",
                        "confidence": 0.99,
                        "warnings": [],
                    },
                    {
                        "segmentIndex": 1,
                        "speakerId": "张三",
                        "emotion": "neutral",
                        "confidence": 0.95,
                        "warnings": [],
                    },
                ],
            }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="院子里很安静。张三说道：“走吧。”",
        language="en",
    ))

    assert len(calls) == 5
    assert "角色固定音色设计器" in calls[1]["messages"][0]["content"]
    assert "动态演绎导演" in calls[4]["messages"][0]["content"]
    system_prompts = [call["messages"][0]["content"] for call in calls]
    user_prompt = json.loads(calls[0]["messages"][1]["content"])
    assert "角色身份分析器" in system_prompts[0]
    assert "角色固定音色设计器" in system_prompts[1]
    assert "说话人归属分析器" in system_prompts[2]
    assert "表达标注分析器" in system_prompts[3]
    assert user_prompt["language"] == "zh"


def test_chinese_analysis_uses_novel_prompt_and_full_source_context():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "characters": [{
                                    "id": "张三",
                                    "canonicalName": "张三",
                                    "aliases": ["老张"],
                                    "gender": "male",
                                    "ageClass": "adult",
                                    "confidence": 0.95,
                                }],
                                "segmentAnnotations": [
                                    {
                                        "segmentIndex": 0,
                                        "speakerId": "narrator",
                                        "emotion": "neutral",
                                        "confidence": 0.99,
                                        "warnings": [],
                                    },
                                    {
                                        "segmentIndex": 1,
                                        "speakerId": "张三",
                                        "emotion": "neutral",
                                        "confidence": 0.95,
                                        "warnings": [],
                                    },
                                    {
                                        "segmentIndex": 2,
                                        "speakerId": "narrator",
                                        "emotion": "neutral",
                                        "confidence": 0.99,
                                        "warnings": [],
                                    },
                                ],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        ),
        transport=transport,
    )
    text = "张三说道：“走吧。”\n李婶没有回答。"
    analyzer.analyze_chapter(
        ChapterAnalysisRequest(
            book_id="book_zh",
            chapter_id="chapter_002",
            text=text,
            language="zh",
            known_characters=[
                CharacterContext(
                    id="张三",
                    canonical_name="张三",
                    aliases=["老张"],
                    gender="male",
                    age_class="young",
                )
            ],
        )
    )

    assert len(calls) == 5
    assert "角色固定音色设计器" in calls[1]["messages"][0]["content"]
    assert "动态演绎导演" in calls[4]["messages"][0]["content"]
    character_prompt = calls[0]["messages"][0]["content"]
    speaker_prompt = calls[2]["messages"][0]["content"]
    delivery_prompt = calls[3]["messages"][0]["content"]
    user_prompt = json.loads(calls[0]["messages"][1]["content"])
    assert "角色身份分析器" in character_prompt
    assert "不要判断每个片段是谁说的" in character_prompt
    assert "说话人归属分析器" in speaker_prompt
    assert "不能判断情绪和语速" in speaker_prompt
    assert "表达标注分析器" in delivery_prompt
    assert "不得修改 speakerId" in delivery_prompt
    assert "场景导演" not in delivery_prompt
    assert "audioPlan" in delivery_prompt
    assert "音频规划" not in character_prompt
    assert user_prompt["chapterText"] == text
    assert user_prompt["language"] == "zh"
    assert user_prompt["segments"][0]["startOffset"] == 0
    assert user_prompt["segments"][0]["endOffset"] > 0
    assert user_prompt["knownCharacters"][0]["id"] == "张三"
    assert user_prompt["knownCharacters"][0]["ageClass"] == "young"
    delivery_input = json.loads(calls[3]["messages"][1]["content"])
    assert delivery_input["speakerAnnotations"][1]["speakerId"] == "张三"


def test_split_analysis_keeps_voice_design_and_direction_in_independent_stages():
    calls = []
    responses = [
        {
            "characters": [{
                "id": "elizabeth",
                "canonicalName": "Elizabeth",
                "aliases": [],
                "gender": "female",
                "ageClass": "adult",
                "confidence": 0.95,
            }],
        },
        {
            "characters": [{
                "id": "elizabeth",
                "voiceDesign": "角色：克制敏锐的成年女性，声线清亮偏冷，咬字利落，句尾收紧。",
            }],
        },
        {
            "segmentAnnotations": [
                {"segmentIndex": 0, "speakerId": "elizabeth", "confidence": 0.95, "warnings": []},
                {"segmentIndex": 1, "speakerId": "narrator", "confidence": 0.99, "warnings": []},
            ],
        },
        {
            "segmentAnnotations": [
                {"segmentIndex": 0, "emotion": "tense", "pace": "fast"},
                {"segmentIndex": 1, "emotion": "neutral", "pace": "normal"},
            ],
        },
        {
            "directions": [
                {"segmentIndex": 0, "direction": "中等偏快，句首短暂停顿，在“Come in”上加强重音。"},
                {"segmentIndex": 1, "direction": "语速平稳，保持清晰克制的叙述节奏。"},
            ],
        },
    ]

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        response = responses[len(calls) - 1]
        return {"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    result = analyzer.analyze_chapter(
        ChapterAnalysisRequest(
            book_id="book_en",
            chapter_id="chapter_001",
            text='"Come in," Elizabeth said.',
            language="en",
        )
    )

    assert len(calls) == 5
    assert result.characters[0].voice_design.startswith("角色：克制敏锐")
    assert result.voice_directions[0].startswith("中等偏快")
    assert "voiceDesign" not in json.loads(calls[0]["messages"][1]["content"])
    assert "independent fixed voice-design stage" in calls[1]["messages"][0]["content"]
    assert "independent dynamic performance-directing stage" in calls[4]["messages"][0]["content"]


def test_analysis_preserves_generic_anonymous_scene_roles_as_separate_characters():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [
                {
                    "id": "茶客甲",
                    "canonicalName": "茶客甲",
                    "aliases": [],
                    "gender": "unknown",
                    "ageClass": "unknown",
                    "confidence": 0.7,
                },
                {
                    "id": "茶客乙",
                    "canonicalName": "茶客乙",
                    "aliases": [],
                    "gender": "unknown",
                    "ageClass": "unknown",
                    "confidence": 0.7,
                },
            ],
            "segmentAnnotations": [
                {
                    "segmentIndex": 0,
                    "speakerId": "茶客甲",
                    "emotion": "neutral",
                    "confidence": 0.7,
                    "warnings": [],
                },
                {
                    "segmentIndex": 1,
                    "speakerId": "茶客乙",
                    "emotion": "neutral",
                    "confidence": 0.7,
                    "warnings": [],
                },
            ],
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    result = analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="“白幡是做什么？”\n“官老爷都系白腰带？”",
        language="zh",
    ))

    assert [character.canonical_name for character in result.characters] == [
        "茶客甲",
        "茶客乙",
    ]
    assert [annotation.speaker_id for annotation in result.segment_annotations] == [
        "茶客甲",
        "茶客乙",
    ]


def test_split_speaker_prompt_preserves_semantic_anonymous_turn_example():
    from audiobook_worker.llm_stages import _SPEAKER_PROMPTS

    prompt = _SPEAKER_PROMPTS["zh"]

    assert "语义承接" in prompt
    assert "甲甲乙甲乙" in prompt
    assert "甲乙甲乙甲" in prompt


def test_split_prompts_reuse_legacy_rules_by_stage():
    from audiobook_worker.llm_stages import (
        _CHARACTER_PROMPTS,
        _DELIVERY_PROMPTS,
        _SPEAKER_PROMPTS,
        _VOICE_DIRECTION_PROMPTS,
        _VOICE_DESIGN_PROMPTS,
    )

    assert "跨章节角色身份规则" in _CHARACTER_PROMPTS["zh"]
    assert "只要一个人物可能是名册中的已有角色" in _CHARACTER_PROMPTS["zh"]
    assert "以上旧角色规则只用于继承身份、称谓、性别、年龄和角色一致性约束" in _VOICE_DESIGN_PROMPTS["zh"]
    assert "speaker_hint_conflict" in _SPEAKER_PROMPTS["zh"]
    assert "quoted_material" in _SPEAKER_PROMPTS["zh"]
    assert "语速判断标准" in _DELIVERY_PROMPTS["zh"]
    assert "teasing" in _DELIVERY_PROMPTS["zh"]
    assert "语速判断标准" in _VOICE_DIRECTION_PROMPTS["zh"]
    assert "Cross-chapter identity rules" in _CHARACTER_PROMPTS["en"]
    assert "pace" in _DELIVERY_PROMPTS["en"]


def test_voice_design_prompt_is_backend_neutral_and_stable_only():
    from audiobook_worker.llm_stages import _VOICE_DESIGN_PROMPTS

    for prompt in _VOICE_DESIGN_PROMPTS.values():
        assert "MiMo V2.5" not in prompt
        assert "voiceDesign" in prompt
        assert "temporary" in prompt.lower() or "临时" in prompt
        assert "post-processing" in prompt or "后期处理" in prompt
        assert "prompt syntax" in prompt or "提示词语法" in prompt


def test_english_analysis_keeps_english_system_prompt():
    calls = []

    def transport(url, headers, payload, timeout_seconds):
        calls.append(payload)
        return {
            "choices": [{"message": {"content": json.dumps({
                "characters": [],
                "segmentAnnotations": [{
                    "segmentIndex": 0,
                    "speakerId": "narrator",
                    "emotion": "neutral",
                    "confidence": 0.99,
                    "warnings": [],
                }],
            })}}]
        }

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
        ),
        transport=transport,
    )
    analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_en",
        chapter_id="chapter_001",
        text="The room was quiet.",
        language="en",
    ))

    assert len(calls) == 4
    assert "dynamic performance" in calls[3]["messages"][0]["content"]
    assert "character-identity stage" in calls[0]["messages"][0]["content"]
    assert "speaker-attribution stage" in calls[1]["messages"][0]["content"]
    assert "delivery-annotation stage" in calls[2]["messages"][0]["content"]
    assert "Do not assign speakers to segments" in calls[0]["messages"][0]["content"]
    assert "do not create or modify characters" in calls[1]["messages"][0]["content"]


def test_analysis_excludes_reserved_narrator_from_discovered_characters():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [{
                "id": "narrator",
                "canonicalName": "旁白",
                "aliases": [],
                "gender": "neutral",
                "ageClass": "adult",
                "confidence": 1.0,
            }],
            "segmentAnnotations": [{
                "segmentIndex": 0,
                "speakerId": "narrator",
                "emotion": "neutral",
                "confidence": 1.0,
                "warnings": [],
            }],
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    result = analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="院子里很安静。",
        language="zh",
    ))

    assert result.characters == []
    assert result.segment_annotations[0].speaker_id == "narrator"


def test_analysis_rejects_missing_duplicate_and_out_of_range_segment_annotations():
    invalid_annotations = [
        [{"segmentIndex": 0, "speakerId": "narrator"}],
        [
            {"segmentIndex": 0, "speakerId": "narrator"},
            {"segmentIndex": 0, "speakerId": "narrator"},
        ],
        [
            {"segmentIndex": 0, "speakerId": "narrator"},
            {"segmentIndex": 2, "speakerId": "narrator"},
        ],
    ]

    for annotations in invalid_annotations:
        def transport(url, headers, payload, timeout_seconds, annotations=annotations):
            return {"choices": [{"message": {"content": json.dumps({
                "characters": [],
                "segmentAnnotations": annotations,
            })}}]}

        analyzer = OpenAICompatibleAnalyzer(
            OpenAICompatibleConfig(
                provider="deepseek",
                api_key="test-key",
                base_url="https://api.deepseek.com/v1",
                model="deepseek-v4-flash",
                max_retries=1,
            ),
            transport=transport,
        )

        with pytest.raises(RuntimeError, match="segment annotations"):
            analyzer.analyze_chapter(ChapterAnalysisRequest(
                book_id="book_en",
                chapter_id="chapter_001",
                text='He waited. "Come in."',
                language="en",
            ))


def test_analysis_reuses_known_character_id_for_returned_alias():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [{
                "id": "老张",
                "canonicalName": "老张",
                "aliases": [],
                "gender": "male",
                "ageClass": "adult",
                "confidence": 0.91,
            }],
            "segmentAnnotations": [
                {
                    "segmentIndex": 0,
                    "speakerId": "narrator",
                    "emotion": "neutral",
                    "confidence": 0.99,
                    "warnings": [],
                },
                {
                    "segmentIndex": 1,
                    "speakerId": "老张",
                    "emotion": "neutral",
                    "confidence": 0.91,
                    "warnings": [],
                },
            ],
        }, ensure_ascii=False)}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    result = analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_002",
        text="老张说道：“走吧。”",
        language="zh",
        known_characters=[CharacterContext(
            id="char_zhang_jianguo",
            canonical_name="张建国",
            aliases=["老张"],
            gender="male",
        )],
    ))

    assert result.characters[0].id == "char_zhang_jianguo"
    assert result.characters[0].canonical_name == "张建国"
    assert "老张" in result.characters[0].aliases
    assert result.segment_annotations[1].speaker_id == "char_zhang_jianguo"


def test_analysis_rejects_unknown_speaker_reference():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": json.dumps({
            "characters": [],
            "segmentAnnotations": [{
                "segmentIndex": 0,
                "speakerId": "not_a_character",
                "emotion": "neutral",
                "confidence": 0.8,
                "warnings": [],
            }],
        })}}]}

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="unknown speakerId"):
        analyzer.analyze_chapter(ChapterAnalysisRequest(
            book_id="book_en",
            chapter_id="chapter_001",
            text="Silence.",
            language="en",
        ))


def test_split_analysis_resumes_from_saved_upstream_stages():
    calls = []
    fail_delivery = True

    def response(payload):
        content = payload
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    def transport(url, headers, payload, timeout_seconds):
        nonlocal fail_delivery
        calls.append(payload)
        stage_index = len(calls)
        if stage_index == 3 and fail_delivery:
            raise RuntimeError("delivery provider temporarily unavailable")
        if stage_index in {1, 2}:
            if stage_index == 1:
                return response({"characters": []})
            return response({
                "segmentAnnotations": [{
                    "segmentIndex": 0,
                    "speakerId": "narrator",
                    "confidence": 0.9,
                    "warnings": [],
                }]
            })
        return response({
            "segmentAnnotations": [{
                "segmentIndex": 0,
                "emotion": "tense",
                "pace": "fast",
            }]
        })

    analyzer = OpenAICompatibleAnalyzer(
        OpenAICompatibleConfig(
            provider="deepseek",
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            max_retries=1,
        ),
        transport=transport,
    )
    saved = {}

    with pytest.raises(RuntimeError, match="delivery provider"):
        analyzer.analyze_chapter(ChapterAnalysisRequest(
            book_id="book_zh",
            chapter_id="chapter_001",
            text="门外传来脚步声。",
            language="zh",
            stage_callback=lambda stage, payload: saved.__setitem__(stage, payload),
        ))

    assert set(saved) == {"characters", "voice_design", "speakers"}
    fail_delivery = False
    result = analyzer.analyze_chapter(ChapterAnalysisRequest(
        book_id="book_zh",
        chapter_id="chapter_001",
        text="门外传来脚步声。",
        language="zh",
        cached_stages=saved,
        resume_from_stage="delivery",
    ))

    assert len(calls) == 5
    assert result.segment_annotations[0].emotion == "tense"
    assert result.segment_annotations[0].pace == "fast"
