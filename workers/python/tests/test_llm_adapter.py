import json

import pytest

from audiobook_worker.llm import (
    CharacterContext,
    ChapterAnalysisRequest,
    MockLLMAnalyzer,
    OpenAICompatibleAnalyzer,
    OpenAICompatibleConfig,
    analyzer_from_models_config,
    resolve_model_from_config,
)


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
    assert result.characters[0].canonical_name == "Elizabeth"
    assert result.segment_annotations[0].emotion == "happy"
    assert result.segment_annotations[0].pace == "fast"
    assert result.segment_annotations[1].pace == "fast"


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

    system_prompt = calls[0]["messages"][0]["content"]
    user_prompt = json.loads(calls[0]["messages"][1]["content"])
    assert "中文有声书剧本分析器" in system_prompt
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

    system_prompt = calls[0]["messages"][0]["content"]
    user_prompt = json.loads(calls[0]["messages"][1]["content"])
    assert "中文有声书剧本分析器" in system_prompt
    assert "预切分" in system_prompt
    assert "内心独白" in system_prompt
    assert "书名、信件、公告、诗句" in system_prompt
    assert "不得仅凭对白轮流出现机械分配说话人" in system_prompt
    assert "匿名连续对白" in system_prompt
    assert "可独立配音的场景角色" in system_prompt
    assert "情绪对语速的默认倾向" in system_prompt
    assert "茶肆/街头/集市匿名闲谈" in system_prompt
    assert "不能因一句警告将整场对白改为 slow" in system_prompt
    assert "whispering 只说明音量" in system_prompt
    assert "teasing" in system_prompt
    assert "废物，喂狗么" in system_prompt
    assert "哟，还挺乖" in system_prompt
    assert "亲属身份是重要证据" in system_prompt
    assert "全书角色身份消歧" in system_prompt
    assert "确认不匹配后才允许新建角色" in system_prompt
    assert "泛称本身不是唯一人物" in system_prompt
    assert "不能为了迎合预切分提示" in system_prompt
    assert "最终自检" in system_prompt
    assert user_prompt["chapterText"] == text
    assert user_prompt["language"] == "zh"
    assert user_prompt["segments"][0]["startOffset"] == 0
    assert user_prompt["segments"][0]["endOffset"] > 0
    assert user_prompt["knownCharacters"][0]["id"] == "张三"
    assert user_prompt["knownCharacters"][0]["ageClass"] == "young"


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

    system_prompt = calls[0]["messages"][0]["content"]
    assert "audiobook script analyst" in system_prompt
    assert "中文有声书剧本分析器" not in system_prompt
    assert "Pace means delivery speed, not volume" in system_prompt
    assert "ordinary group chatter" in system_prompt
    assert "Cross-chapter identity rules" in system_prompt
    assert "temporary model label" in system_prompt
    assert "mechanical A-B-A-B alternation" in system_prompt
    assert "Final self-check" in system_prompt


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
