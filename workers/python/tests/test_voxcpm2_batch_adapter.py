from pathlib import Path
from types import SimpleNamespace

import torch

from audiobook_worker.voxcpm2_batch_adapter import VoxCPM2BatchAdapter


_HIDDEN = 4
_PATCH_SIZE = 2


class _FakeKVCache:
    def __init__(self, layer_count: int, batch_size: int, max_length: int) -> None:
        self.max_length = max_length
        self.current_length = 0
        self._layers = [
            (
                torch.zeros(batch_size, 1, max_length, _HIDDEN),
                torch.zeros(batch_size, 1, max_length, _HIDDEN),
            )
            for _ in range(layer_count)
        ]

    def get_layer_cache(self, index: int):
        return self._layers[index]

    def fill_caches(self, caches) -> None:
        self.current_length = int(caches[0][0].shape[2])
        for target, source in zip(self._layers, caches, strict=True):
            for target_tensor, source_tensor in zip(target, source, strict=True):
                target_tensor.zero_()
                target_tensor[:, :, : self.current_length, :] = source_tensor

    def step(self) -> int:
        position = self.current_length
        if position >= self.max_length:
            raise RuntimeError("fake KV cache is full")
        self.current_length += 1
        return position


class _FakeAttention:
    num_heads = 1
    num_key_value_heads = 1
    head_dim = _HIDDEN

    def __init__(self) -> None:
        self.q_proj = torch.nn.Identity()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()
        self.o_proj = torch.nn.Identity()


class _FakeLayer:
    use_mup = False
    scale_depth = 1.0
    num_hidden_layers = 1

    def __init__(self) -> None:
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = _FakeAttention()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Identity()


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.embed_tokens = torch.nn.Embedding(256, _HIDDEN)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(256 * _HIDDEN, dtype=torch.float32).reshape(256, _HIDDEN) / 1000
            )
        self.layers = [_FakeLayer()]
        self.norm = torch.nn.Identity()
        self.rope_emb = None
        self.kv_cache: _FakeKVCache | None = None
        self.setup_batches: list[int] = []
        self.setup_lengths: list[int] = []

    def setup_cache(self, batch_size: int, max_length: int, _device, _dtype) -> None:
        self.setup_batches.append(batch_size)
        self.setup_lengths.append(max_length)
        self.kv_cache = _FakeKVCache(len(self.layers), batch_size, max_length)


class _FakeFeatureEncoder:
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        averaged = values.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        return averaged.repeat(1, 1, _HIDDEN)


class _FirstHalf:
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., :_HIDDEN]


class _FakeStopHead:
    """Stop the first item one patch before the other active items."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        flags = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        if self.calls >= 3:
            flags[0] = 1
        if self.calls >= 4:
            flags[:] = 1
        logits = torch.zeros(values.shape[0], 2, dtype=values.dtype, device=values.device)
        logits[:, 0] = 1 - flags
        logits[:, 1] = flags
        self.calls += 1
        return logits


class _FakeFeatureDecoder:
    in_channels = _HIDDEN

    def solve_euler(self, x, t_span, mu, cond, cfg_value, use_cfg_zero_star):
        del t_span, cond, cfg_value, use_cfg_zero_star
        return x * 0 + mu[:, : self.in_channels].unsqueeze(-1)


class _FakeAudioVAE:
    latent_dim = _HIDDEN

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :1, :]


class _FakeTtsModel:
    device = "cpu"
    patch_size = _PATCH_SIZE
    audio_start_token = 101

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            dtype="float32",
            max_length=512,
            lm_config=SimpleNamespace(use_mup=False, scale_emb=1.0),
        )
        self.base_lm = _FakeLanguageModel()
        self.residual_lm = _FakeLanguageModel()
        self.feat_encoder = _FakeFeatureEncoder()
        self.feat_decoder = _FakeFeatureDecoder()
        self.audio_vae = _FakeAudioVAE()
        self.enc_to_lm_proj = torch.nn.Identity()
        self.fsq_layer = torch.nn.Identity()
        self.lm_to_dit_proj = torch.nn.Identity()
        self.res_to_dit_proj = torch.nn.Identity()
        self.fusion_concat_proj = _FirstHalf()
        self.stop_proj = torch.nn.Identity()
        self.stop_actn = torch.nn.Identity()
        self.stop_head = _FakeStopHead()
        self.prompt_cache_calls: list[str] = []

    def text_tokenizer(self, text: str) -> list[int]:
        return [10 + (ord(character) % 80) for character in text if not character.isspace()] or [10]

    def build_prompt_cache(self, *, reference_wav_path: str):
        self.prompt_cache_calls.append(reference_wav_path)
        amplitude = 0.2 if reference_wav_path.endswith("profile-a.wav") else 0.6
        return {
            "mode": "reference",
            "ref_audio_feat": torch.full((2, _PATCH_SIZE, _HIDDEN), amplitude),
        }

    def _make_ref_prefix(self, ref_feat: torch.Tensor, _device):
        ref_length = int(ref_feat.shape[0])
        zero = torch.zeros((1, _PATCH_SIZE, _HIDDEN), dtype=torch.float32)
        tokens = torch.cat(
            (
                torch.tensor([103], dtype=torch.long),
                torch.zeros(ref_length, dtype=torch.long),
                torch.tensor([104], dtype=torch.long),
            )
        )
        features = torch.cat((zero, ref_feat, zero), dim=0)
        text_mask = torch.cat(
            (
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(ref_length, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            )
        )
        audio_mask = torch.cat(
            (
                torch.tensor([0], dtype=torch.int32),
                torch.ones(ref_length, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            )
        )
        return tokens, features, text_mask, audio_mask


def _adapter() -> tuple[VoxCPM2BatchAdapter, _FakeTtsModel]:
    model = _FakeTtsModel()
    return VoxCPM2BatchAdapter(SimpleNamespace(tts_model=model), validate_upstream=False), model


def _segment(index: int, reference_path: Path) -> dict[str, object]:
    return {
        "id": f"seg_{index:04d}",
        "text": "短句" if index == 0 else "长度不同的测试台词。" * (index + 1),
        "delivery": "自然克制，语速适中",
        "referenceWavPath": str(reference_path),
        "cacheSignature": f"signature-{index}",
    }


def test_adapter_batches_one_two_and_four_items_with_independent_stop_and_mapping(tmp_path: Path):
    profile_a = tmp_path / "profile-a.wav"
    profile_b = tmp_path / "profile-b.wav"
    profile_a.write_bytes(b"a")
    profile_b.write_bytes(b"b")

    for batch_size in (1, 2, 4):
        adapter, model = _adapter()
        items = [
            _segment(index, profile_a if index % 2 == 0 else profile_b)
            for index in range(batch_size)
        ]
        results = adapter.generate_batch(items)

        assert [result.segment_id for result in results] == [item["id"] for item in items]
        assert [result.generated_patches for result in results] == [4] + [5] * (batch_size - 1)
        assert all(result.waveform.size > 0 for result in results)
        assert model.base_lm.setup_batches == [batch_size]
        assert model.residual_lm.setup_batches == [batch_size]
        assert all(length < model.config.max_length for length in model.base_lm.setup_lengths)
        assert model.base_lm.setup_lengths == model.residual_lm.setup_lengths
        expected_profiles = {str(profile_a)}
        if batch_size > 1:
            expected_profiles.add(str(profile_b))
        assert set(model.prompt_cache_calls) == expected_profiles


def test_adapter_left_pads_mixed_lengths_and_masks_the_padding(tmp_path: Path):
    profile_a = tmp_path / "profile-a.wav"
    profile_b = tmp_path / "profile-b.wav"
    profile_a.write_bytes(b"a")
    profile_b.write_bytes(b"b")
    adapter, _model = _adapter()
    prepared = [
        adapter._prepare_item(_segment(0, profile_a)),
        adapter._prepare_item(_segment(3, profile_b)),
    ]

    tokens, text_mask, _features, audio_mask, starts = adapter._collate(prepared)
    assert int(starts[0]) > 0
    assert int(starts[1]) == 0
    assert torch.all(tokens[0, : starts[0]] == 0)
    assert torch.all(text_mask[0, : starts[0]] == 0)
    assert torch.all(audio_mask[0, : starts[0]] == 0)
    attention_mask = adapter._prefill_attention_mask(starts, int(tokens.shape[1]))
    assert not bool(attention_mask[0, 0, int(starts[0]), int(starts[0]) - 1])
    assert bool(attention_mask[0, 0, -1, -1])


def test_adapter_sizes_kv_cache_from_the_current_batch_and_disables_autograd(tmp_path: Path):
    profile_a = tmp_path / "profile-a.wav"
    profile_b = tmp_path / "profile-b.wav"
    profile_a.write_bytes(b"a")
    profile_b.write_bytes(b"b")
    adapter, model = _adapter()
    items = [_segment(0, profile_a), _segment(3, profile_b)]
    prepared = [adapter._prepare_item(item) for item in items]
    expected_cache_length = max(int(item.text_token.shape[0]) for item in prepared) + max(
        item.maximum_steps for item in prepared
    )
    inference_mode_states: list[bool] = []
    original_run = adapter._run_batched_inference

    def inspect_inference_mode(*args, **kwargs):
        inference_mode_states.append(torch.is_inference_mode_enabled())
        return original_run(*args, **kwargs)

    adapter._run_batched_inference = inspect_inference_mode
    adapter.generate_batch(items)

    assert model.base_lm.setup_lengths == [expected_cache_length]
    assert model.residual_lm.setup_lengths == [expected_cache_length]
    assert inference_mode_states == [True]


def test_adapter_reuses_distinct_reference_profiles_and_only_invalidates_rewritten_one(tmp_path: Path):
    profile_a = tmp_path / "profile-a.wav"
    profile_b = tmp_path / "profile-b.wav"
    profile_a.write_bytes(b"a")
    profile_b.write_bytes(b"b")
    adapter, model = _adapter()

    adapter._reference_prompt_cache(profile_a)
    adapter._reference_prompt_cache(profile_b)
    adapter._reference_prompt_cache(profile_a)
    assert model.prompt_cache_calls == [str(profile_a), str(profile_b)]

    profile_a.write_bytes(b"rewritten-profile-a")
    adapter._reference_prompt_cache(profile_a)
    adapter._reference_prompt_cache(profile_b)
    assert model.prompt_cache_calls == [str(profile_a), str(profile_b), str(profile_a)]
