"""Project-owned true tensor batching for the pinned VoxCPM2 runtime.

The upstream package exposes a single-item public ``generate`` API even
though its lower-level language, diffusion, and VAE modules already accept a
batch dimension.  This adapter intentionally lives outside the installed
virtual environment: it validates the known upstream source shape, collates
reference-cloning requests with masks, allocates the model KV caches for the
active batch size, and tracks stop/length/output state independently for every
item.

Only the isolated resident service imports and executes this module.  The
regular audiobook worker can safely import its version constant but never
constructs an adapter or imports :mod:`voxcpm`.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as functional

if __package__:
    from .voxcpm2_service_protocol import BATCH_ADAPTER_VERSION as BATCH_ADAPTER_VERSION
else:  # pragma: no cover - imported by the isolated direct-script service.
    from voxcpm2_service_protocol import BATCH_ADAPTER_VERSION as BATCH_ADAPTER_VERSION


MAX_BATCH_SIZE = 4
_MAX_REFERENCE_PROMPT_CACHES = 64
# The installed ``voxcpm/model/voxcpm2.py`` source is deliberately pinned. A
# package update that changes private cache or inference details must use the
# legacy runner until this adapter and its tests are reviewed against it.
_SUPPORTED_VOXCPM2_SOURCE_SHA256 = (
    "9c4f1e08495e727ce339e5da93347237b81f386398a783b79c66eb206c3397f6"
)
_SUPPORTED_MINICPM4_SOURCE_SHA256 = (
    "0733349295d13b2ec57e979058fa64becec821b9bd475e188eb4033fc016d2d2"
)
_SUPPORTED_STATIC_KV_CACHE_SOURCE_SHA256 = (
    "644acd3bf7cf8313f2ba285f68c41967fcedb57e5f5d8f38a883acc385ee030d"
)
_SUPPORTED_UNIFIED_CFM_SOURCE_SHA256 = (
    "e4535337024cf44b6888eccfd39ccbe98684e302dd1836e05f940b4d626dcfdf"
)
_SOURCE_SHAPE_MARKERS = (
    "self.base_lm.setup_cache(1",
    "has_continuation_audio = feat_mask[0, -1].item() == 1",
    "argmax(dim=-1)[0].cpu().item()",
    "generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()",
)
_MINICPM4_SOURCE_SHAPE_MARKERS = (
    "class MiniCPMModel",
    "def forward_step(",
    "self.kv_cache.get_layer_cache(i)",
)
_STATIC_KV_CACHE_SOURCE_SHAPE_MARKERS = (
    "class StaticKVCache",
    "def fill_caches",
    "self.current_length = kv_caches[0][0].size(2)",
)
_UNIFIED_CFM_SOURCE_SHAPE_MARKERS = (
    "class UnifiedCFM",
    "def solve_euler(",
    "use_cfg_zero_star",
)


class VoxCPM2BatchAdapterError(RuntimeError):
    """The installed runtime cannot safely execute a requested tensor batch."""


class VoxCPM2BatchAdapterIncompatibleError(VoxCPM2BatchAdapterError):
    """The pinned upstream private implementation no longer has the expected shape."""


@dataclass(frozen=True)
class BatchAudioResult:
    """One independent generated waveform, keyed by the original segment id."""

    segment_id: str
    waveform: Any
    generated_patches: int


@dataclass(frozen=True)
class _PreparedItem:
    raw: dict[str, Any]
    segment_id: str
    target_text: str
    target_token_length: int
    maximum_steps: int
    seed: int
    text_token: torch.Tensor
    text_mask: torch.Tensor
    audio_feat: torch.Tensor
    audio_mask: torch.Tensor


def _controlled_text(instruction: object, text: object) -> str:
    normalized_instruction = " ".join(str(instruction or "").split())
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise VoxCPM2BatchAdapterError("VoxCPM2 batch item text is required.")
    return f"({normalized_instruction}){normalized_text}" if normalized_instruction else normalized_text


def _seed_from_item(item: dict[str, Any], target_text: str) -> int:
    """Derive a membership-independent CPU RNG seed for one diffusion item."""

    identity = "\u0000".join(
        (
            str(item.get("id") or ""),
            target_text,
            str(item.get("referenceWavPath") or ""),
            str(item.get("cacheSignature") or ""),
        )
    )
    # ``torch.Generator`` accepts a signed 64-bit seed.  Using a CPU generator
    # below makes each item's diffusion noise independent of its batch order
    # and works on MPS, which does not expose all generator APIs.
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") % (
        2**63 - 1
    )


def _torch_dtype(tts_model: Any) -> torch.dtype:
    configured = str(getattr(getattr(tts_model, "config", None), "dtype", "float32"))
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[configured.casefold()]
    except KeyError as error:
        raise VoxCPM2BatchAdapterIncompatibleError(
            f"Unsupported VoxCPM2 model dtype: {configured}"
        ) from error


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embedding: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_embedding is None:
        return query, key
    cos, sin = position_embedding
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


def _attention_prefill(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embedding: tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """The upstream attention implementation plus an explicit pad mask."""

    batch_size, query_length, _ = hidden_states.size()
    query = attention.q_proj(hidden_states)
    key = attention.k_proj(hidden_states)
    value = attention.v_proj(hidden_states)
    query = query.view(batch_size, query_length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(
        batch_size,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    value = value.view(
        batch_size,
        query_length,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    query, key = _apply_rotary(query, key, position_embedding)
    output = functional.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )
    output = output.transpose(1, 2).contiguous().reshape(
        batch_size,
        query_length,
        attention.num_heads * attention.head_dim,
    )
    return attention.o_proj(output), (key, value)


def _attention_step(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embedding: tuple[torch.Tensor, torch.Tensor] | None,
    position: int,
    key_valid_mask: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """One batched autoregressive step with per-item left-pad exclusion."""

    batch_size, _ = hidden_states.size()
    query = attention.q_proj(hidden_states)
    key = attention.k_proj(hidden_states)
    value = attention.v_proj(hidden_states)
    query = query.view(batch_size, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch_size, 1, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch_size, 1, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    query, key = _apply_rotary(query, key, position_embedding)
    key_cache, value_cache = kv_cache
    # Keep the singleton sequence axis.  Integer indexing would collapse it
    # to ``[B, heads, dim]`` while the new key/value retain
    # ``[B, heads, 1, dim]``; that happens to be hidden by B=1 in the upstream
    # runner but rejects a true multi-item cache write.
    key_cache[:, :, position : position + 1, :] = key
    value_cache[:, :, position : position + 1, :] = value
    output = functional.scaled_dot_product_attention(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        attn_mask=key_valid_mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )
    output = output.transpose(1, 2).contiguous().reshape(
        batch_size,
        attention.num_heads * attention.head_dim,
    )
    return attention.o_proj(output)


def _decoder_prefill(
    layer: Any,
    hidden_states: torch.Tensor,
    position_embedding: tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states, cache = _attention_prefill(
        layer.self_attn,
        hidden_states,
        position_embedding,
        attention_mask,
    )
    if layer.use_mup:
        hidden_states = residual + hidden_states * (
            layer.scale_depth / (layer.num_hidden_layers**0.5)
        )
    else:
        hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if layer.use_mup:
        hidden_states = residual + hidden_states * (
            layer.scale_depth / (layer.num_hidden_layers**0.5)
        )
    else:
        hidden_states = residual + hidden_states
    return hidden_states, cache


def _decoder_step(
    layer: Any,
    hidden_states: torch.Tensor,
    position_embedding: tuple[torch.Tensor, torch.Tensor] | None,
    position: int,
    key_valid_mask: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _attention_step(
        layer.self_attn,
        hidden_states,
        position_embedding,
        position,
        key_valid_mask,
        kv_cache,
    )
    if layer.use_mup:
        hidden_states = residual + hidden_states * (
            layer.scale_depth / (layer.num_hidden_layers**0.5)
        )
    else:
        hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    if layer.use_mup:
        return residual + hidden_states * (layer.scale_depth / (layer.num_hidden_layers**0.5))
    return residual + hidden_states


class VoxCPM2BatchAdapter:
    """Batch reference-cloning segments on one resident VoxCPM2 model."""

    def __init__(self, pipeline: Any, *, validate_upstream: bool = True) -> None:
        self.pipeline = pipeline
        self.tts_model = getattr(pipeline, "tts_model", None)
        if self.tts_model is None:
            raise VoxCPM2BatchAdapterIncompatibleError("VoxCPM2 pipeline has no tts_model.")
        # Reference features are CPU tensors.  Keep distinct character/profile
        # references warm during a batch, but cap the cache so a long book with
        # many one-off roles cannot grow the resident process indefinitely.
        self._prompt_caches: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
        self._cached_cache_shape: tuple[int, int] | None = None
        if validate_upstream:
            self.assert_supported_upstream(self.tts_model)

    @classmethod
    def assert_supported_upstream(cls, tts_model: Any) -> None:
        """Fail closed when a virtual-environment update changes private APIs."""

        if tts_model.__class__.__name__ != "VoxCPM2Model":
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Expected VoxCPM2Model, got {tts_model.__class__.__name__}."
            )
        required = (
            "base_lm",
            "residual_lm",
            "feat_encoder",
            "feat_decoder",
            "audio_vae",
            "_make_ref_prefix",
            "build_prompt_cache",
        )
        missing = [name for name in required if not hasattr(tts_model, name)]
        if missing:
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"VoxCPM2 runtime is missing required batch hooks: {', '.join(missing)}"
            )
        private_components = (
            (
                "VoxCPM2Model",
                tts_model,
                _SUPPORTED_VOXCPM2_SOURCE_SHA256,
                _SOURCE_SHAPE_MARKERS,
            ),
            (
                "MiniCPMModel",
                tts_model.base_lm,
                _SUPPORTED_MINICPM4_SOURCE_SHA256,
                _MINICPM4_SOURCE_SHAPE_MARKERS,
            ),
            (
                "StaticKVCache",
                getattr(tts_model.base_lm, "kv_cache", None),
                _SUPPORTED_STATIC_KV_CACHE_SOURCE_SHA256,
                _STATIC_KV_CACHE_SOURCE_SHAPE_MARKERS,
            ),
            (
                "UnifiedCFM",
                tts_model.feat_decoder,
                _SUPPORTED_UNIFIED_CFM_SOURCE_SHA256,
                _UNIFIED_CFM_SOURCE_SHAPE_MARKERS,
            ),
        )
        for expected_name, component, expected_hash, markers in private_components:
            cls._assert_private_source_pin(
                expected_name,
                component,
                expected_hash=expected_hash,
                markers=markers,
            )

    @staticmethod
    def _assert_private_source_pin(
        expected_name: str,
        component: Any,
        *,
        expected_hash: str,
        markers: tuple[str, ...],
    ) -> None:
        if component is None or component.__class__.__name__ != expected_name:
            actual_name = component.__class__.__name__ if component is not None else "missing"
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Expected {expected_name}, got {actual_name}."
            )
        source_path_text = inspect.getsourcefile(component.__class__)
        if not source_path_text:
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Cannot locate {expected_name} implementation source."
            )
        source_path = Path(source_path_text)
        try:
            source = source_path.read_bytes()
        except OSError as error:
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Cannot read {expected_name} implementation source."
            ) from error
        if hashlib.sha256(source).hexdigest() != expected_hash:
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Installed {expected_name} source differs from the supported batch-adapter pin."
            )
        decoded = source.decode("utf-8", errors="replace")
        missing_markers = [marker for marker in markers if marker not in decoded]
        if missing_markers:
            raise VoxCPM2BatchAdapterIncompatibleError(
                f"Installed {expected_name} source no longer has the expected inference shape."
            )

    @torch.inference_mode()
    def generate_batch(self, items: list[dict[str, Any]]) -> list[BatchAudioResult]:
        """Generate independent reference-cloning items in one tensor batch."""

        if not items:
            return []
        if len(items) > MAX_BATCH_SIZE:
            raise VoxCPM2BatchAdapterError(
                f"VoxCPM2 batch is limited to {MAX_BATCH_SIZE} items."
            )
        prepared = [self._prepare_item(item) for item in items]
        maximum_steps = max(item.maximum_steps for item in prepared)
        text_token, text_mask, audio_feat, audio_mask, starts = self._collate(prepared)
        initial_length = int(text_token.shape[1])
        maximum_length = int(getattr(self.tts_model.config, "max_length", 0) or 0)
        required_cache_length = initial_length + maximum_steps
        if maximum_length <= 0 or required_cache_length >= maximum_length:
            raise VoxCPM2BatchAdapterError(
                "VoxCPM2 batch input plus generation length exceeds its KV cache capacity."
            )
        self._configure_cache(len(prepared), required_cache_length)
        generated = self._run_batched_inference(
            prepared,
            text_token,
            text_mask,
            audio_feat,
            audio_mask,
            starts,
        )
        return [
            BatchAudioResult(
                segment_id=item.segment_id,
                waveform=self._decode_generated_features(features),
                generated_patches=int(features.shape[0]),
            )
            for item, features in zip(prepared, generated, strict=True)
        ]

    def _prepare_item(self, raw: dict[str, Any]) -> _PreparedItem:
        segment_id = str(raw.get("id") or "").strip()
        if not segment_id:
            raise VoxCPM2BatchAdapterError("VoxCPM2 batch item has no segment id.")
        target_text = _controlled_text(raw.get("delivery"), raw.get("text"))
        reference_path = Path(str(raw.get("referenceWavPath") or "").strip())
        if not reference_path.is_file():
            raise VoxCPM2BatchAdapterError(
                f"VoxCPM2 reference WAV is unavailable for segment {segment_id}: {reference_path}"
            )
        prompt_cache = self._reference_prompt_cache(reference_path)
        if prompt_cache.get("mode") != "reference":
            raise VoxCPM2BatchAdapterError(
                "The batch adapter only supports reference-cloning segment requests."
            )
        text_token = torch.LongTensor(self.tts_model.text_tokenizer(target_text))
        if text_token.numel() == 0:
            raise VoxCPM2BatchAdapterError(f"VoxCPM2 tokenized an empty segment: {segment_id}")
        text_token = torch.cat(
            (
                text_token,
                torch.tensor([self.tts_model.audio_start_token], dtype=torch.long),
            )
        )
        ref_tokens, ref_features, ref_text_mask, ref_audio_mask = self.tts_model._make_ref_prefix(
            prompt_cache["ref_audio_feat"],
            torch.device("cpu"),
        )
        text_length = int(text_token.shape[0])
        text_features = torch.zeros(
            (
                text_length,
                self.tts_model.patch_size,
                self.tts_model.audio_vae.latent_dim,
            ),
            dtype=torch.float32,
        )
        text_mask = torch.cat(
            (
                ref_text_mask.to(torch.int32),
                torch.ones(text_length, dtype=torch.int32),
            )
        )
        audio_mask = torch.cat(
            (
                ref_audio_mask.to(torch.int32),
                torch.zeros(text_length, dtype=torch.int32),
            )
        )
        combined_token = torch.cat((ref_tokens.to(torch.long), text_token))
        combined_features = torch.cat((ref_features.to(torch.float32), text_features), dim=0)
        target_token_length = len(self.tts_model.text_tokenizer(target_text))
        if target_token_length <= 0:
            raise VoxCPM2BatchAdapterError(f"VoxCPM2 tokenized an empty target: {segment_id}")
        maximum_steps = min(int(target_token_length * 6.0 + 10), 4096)
        return _PreparedItem(
            raw=raw,
            segment_id=segment_id,
            target_text=target_text,
            target_token_length=target_token_length,
            maximum_steps=max(3, maximum_steps),
            seed=_seed_from_item(raw, target_text),
            text_token=combined_token,
            text_mask=text_mask,
            audio_feat=combined_features,
            audio_mask=audio_mask,
        )

    def _reference_prompt_cache(self, reference_path: Path) -> dict[str, Any]:
        stat = reference_path.stat()
        key = (str(reference_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        cached = self._prompt_caches.pop(key, None)
        if cached is not None:
            self._prompt_caches[key] = cached
            return cached
        prompt_cache = self.tts_model.build_prompt_cache(reference_wav_path=str(reference_path))
        if not isinstance(prompt_cache, dict):
            raise VoxCPM2BatchAdapterError("VoxCPM2 returned an invalid reference prompt cache.")
        # A rewritten reference must not leave an obsolete tensor in the
        # resident process for a later chapter using the same filename, but a
        # different character/profile must remain reusable in this service.
        stale_keys = [
            cache_key
            for cache_key in self._prompt_caches
            if cache_key[0] == key[0]
        ]
        for cache_key in stale_keys:
            self._prompt_caches.pop(cache_key, None)
        self._prompt_caches[key] = prompt_cache
        while len(self._prompt_caches) > _MAX_REFERENCE_PROMPT_CACHES:
            self._prompt_caches.popitem(last=False)
        return prompt_cache

    def _configure_cache(self, batch_size: int, cache_length: int) -> None:
        cache_shape = (batch_size, cache_length)
        if cache_shape == self._cached_cache_shape:
            return
        dtype = _torch_dtype(self.tts_model)
        maximum_length = int(getattr(self.tts_model.config, "max_length", 0) or 0)
        if maximum_length <= 0 or cache_length <= 0 or cache_length >= maximum_length:
            raise VoxCPM2BatchAdapterIncompatibleError("VoxCPM2 has no positive cache length.")
        try:
            self.tts_model.base_lm.setup_cache(
                batch_size,
                cache_length,
                self.tts_model.device,
                dtype,
            )
            self.tts_model.residual_lm.setup_cache(
                batch_size,
                cache_length,
                self.tts_model.device,
                dtype,
            )
        except Exception as error:
            raise VoxCPM2BatchAdapterError(
                f"VoxCPM2 could not allocate a batch-{batch_size} KV cache: {error}"
            ) from error
        self._cached_cache_shape = cache_shape

    def _collate(
        self,
        prepared: Iterable[_PreparedItem],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        items = list(prepared)
        maximum_length = max(int(item.text_token.shape[0]) for item in items)
        tokens: list[torch.Tensor] = []
        text_masks: list[torch.Tensor] = []
        audio_features: list[torch.Tensor] = []
        audio_masks: list[torch.Tensor] = []
        starts: list[int] = []
        for item in items:
            length = int(item.text_token.shape[0])
            padding = maximum_length - length
            starts.append(padding)
            tokens.append(torch.cat((torch.zeros(padding, dtype=torch.long), item.text_token)))
            text_masks.append(
                torch.cat((torch.zeros(padding, dtype=torch.int32), item.text_mask))
            )
            audio_masks.append(
                torch.cat((torch.zeros(padding, dtype=torch.int32), item.audio_mask))
            )
            audio_features.append(
                torch.cat(
                (
                    torch.zeros(
                        (
                            padding,
                            self.tts_model.patch_size,
                            self.tts_model.audio_vae.latent_dim,
                        ),
                        dtype=torch.float32,
                    ),
                    item.audio_feat,
                ),
                dim=0,
                )
            )
        device = torch.device(self.tts_model.device)
        return (
            torch.stack(tokens, dim=0).to(device),
            torch.stack(text_masks, dim=0).to(device),
            torch.stack(audio_features, dim=0).to(device).to(_torch_dtype(self.tts_model)),
            torch.stack(audio_masks, dim=0).to(device),
            torch.tensor(starts, dtype=torch.long, device=device),
        )

    @staticmethod
    def _prefill_attention_mask(starts: torch.Tensor, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=starts.device)
        query_positions = positions.view(1, sequence_length, 1)
        key_positions = positions.view(1, 1, sequence_length)
        start_positions = starts.view(-1, 1, 1)
        query_valid = query_positions >= start_positions
        key_valid = key_positions >= start_positions
        mask = (query_positions >= key_positions) & query_valid & key_valid
        # Invalid left-pad query rows are never consumed, but SDPA still needs
        # at least one allowed key for them to avoid all--inf softmax rows.
        mask |= (~query_valid) & (key_positions == 0)
        return mask.unsqueeze(1)

    @staticmethod
    def _step_attention_mask(
        starts: torch.Tensor,
        cache_length: int,
        position: int,
    ) -> torch.Tensor:
        positions = torch.arange(cache_length, device=starts.device)
        valid = (positions.view(1, -1) >= starts.view(-1, 1)) & (positions <= position)
        return valid[:, None, None, :]

    def _lm_prefill(
        self,
        language_model: Any,
        embeddings: torch.Tensor,
        starts: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        sequence_length = int(embeddings.shape[1])
        if getattr(language_model, "rope_emb", None) is not None:
            positions = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=embeddings.device,
            )
            position_embedding = language_model.rope_emb(positions)
        else:
            position_embedding = None
        attention_mask = self._prefill_attention_mask(starts, sequence_length)
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        hidden_states = embeddings
        for layer in language_model.layers:
            hidden_states, cache = _decoder_prefill(
                layer,
                hidden_states,
                position_embedding,
                attention_mask,
            )
            caches.append(cache)
        return language_model.norm(hidden_states), caches

    def _lm_step(
        self,
        language_model: Any,
        embeddings: torch.Tensor,
        starts: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        if getattr(language_model, "kv_cache", None) is None:
            raise VoxCPM2BatchAdapterIncompatibleError("VoxCPM2 language-model cache is unavailable.")
        if getattr(language_model, "rope_emb", None) is not None:
            position_tensor = torch.tensor([position], dtype=torch.long, device=embeddings.device)
            position_embedding = language_model.rope_emb(position_tensor)
        else:
            position_embedding = None
        cache_length = int(language_model.kv_cache.max_length)
        attention_mask = self._step_attention_mask(starts, cache_length, position)
        hidden_states = embeddings
        for index, layer in enumerate(language_model.layers):
            hidden_states = _decoder_step(
                layer,
                hidden_states,
                position_embedding,
                position,
                attention_mask,
                language_model.kv_cache.get_layer_cache(index),
            )
        return language_model.norm(hidden_states)

    def _diffuse_patch(
        self,
        mu: torch.Tensor,
        condition: torch.Tensor,
        seeds: list[int],
        inference_timesteps: int,
        cfg_value: float,
    ) -> torch.Tensor:
        decoder = self.tts_model.feat_decoder
        noises = []
        for seed in seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            noise = torch.randn(
                (decoder.in_channels, self.tts_model.patch_size),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            noises.append(noise)
        initial = torch.stack(noises, dim=0).to(device=mu.device, dtype=mu.dtype)
        span = torch.linspace(
            1,
            0,
            inference_timesteps + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        span = span + (torch.cos(torch.pi / 2 * span) - 1 + span)
        return decoder.solve_euler(
            x=initial,
            t_span=span,
            mu=mu,
            cond=condition,
            cfg_value=cfg_value,
            use_cfg_zero_star=True,
        ).transpose(1, 2)

    def _run_batched_inference(
        self,
        prepared: list[_PreparedItem],
        text_token: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor,
        starts: torch.Tensor,
    ) -> list[torch.Tensor]:
        batch_size = len(prepared)
        base_model = self.tts_model.base_lm
        residual_model = self.tts_model.residual_lm
        prefill_encoder = getattr(self.tts_model, "_feat_encoder_raw", self.tts_model.feat_encoder)
        feature_embeddings = self.tts_model.enc_to_lm_proj(prefill_encoder(audio_feat))
        scale = (
            self.tts_model.config.lm_config.scale_emb
            if self.tts_model.config.lm_config.use_mup
            else 1.0
        )
        text_embeddings = base_model.embed_tokens(text_token) * scale
        combined = text_mask.unsqueeze(-1) * text_embeddings + audio_mask.unsqueeze(-1) * feature_embeddings
        base_outputs, base_caches = self._lm_prefill(base_model, combined, starts)
        base_model.kv_cache.fill_caches(base_caches)
        base_outputs = self.tts_model.fsq_layer(base_outputs) * audio_mask.unsqueeze(-1) + base_outputs * text_mask.unsqueeze(-1)
        language_hidden = base_outputs[:, -1, :]
        residual_inputs = self.tts_model.fusion_concat_proj(
            torch.cat((base_outputs, audio_mask.unsqueeze(-1) * feature_embeddings), dim=-1)
        )
        residual_outputs, residual_caches = self._lm_prefill(residual_model, residual_inputs, starts)
        residual_model.kv_cache.fill_caches(residual_caches)
        residual_hidden = residual_outputs[:, -1, :]
        prefix_condition = audio_feat[:, -1, ...]
        generated: list[list[torch.Tensor]] = [[] for _ in prepared]
        active = torch.ones(batch_size, dtype=torch.bool, device=text_token.device)
        maximum_steps = torch.tensor(
            [item.maximum_steps for item in prepared],
            dtype=torch.long,
            device=text_token.device,
        )
        for step in range(int(maximum_steps.max().item())):
            projected_language = self.tts_model.lm_to_dit_proj(language_hidden)
            projected_residual = self.tts_model.res_to_dit_proj(residual_hidden)
            diffusion_hidden = torch.cat((projected_language, projected_residual), dim=-1)
            patch = self._diffuse_patch(
                diffusion_hidden,
                prefix_condition.transpose(1, 2).contiguous(),
                [item.seed + step for item in prepared],
                inference_timesteps=10,
                cfg_value=2.0,
            )
            for index in range(batch_size):
                if bool(active[index].item()):
                    generated[index].append(patch[index])
            stop_flags = self.tts_model.stop_head(
                self.tts_model.stop_actn(self.tts_model.stop_proj(language_hidden))
            ).argmax(dim=-1)
            finished = active & (
                ((step > 2) & (stop_flags == 1)) | ((step + 1) >= maximum_steps)
            )
            active = active & ~finished
            if not bool(active.any().item()):
                break
            current_embeddings = self.tts_model.enc_to_lm_proj(
                self.tts_model.feat_encoder(patch.unsqueeze(1))
            )
            base_position = base_model.kv_cache.step()
            language_hidden = self._lm_step(
                base_model,
                current_embeddings[:, 0, :],
                starts,
                base_position,
            ).clone()
            language_hidden = self.tts_model.fsq_layer(language_hidden)
            residual_input = self.tts_model.fusion_concat_proj(
                torch.cat((language_hidden, current_embeddings[:, 0, :]), dim=-1)
            )
            residual_position = residual_model.kv_cache.step()
            residual_hidden = self._lm_step(
                residual_model,
                residual_input,
                starts,
                residual_position,
            ).clone()
            prefix_condition = patch
        outputs: list[torch.Tensor] = []
        for item, patches in zip(prepared, generated, strict=True):
            if not patches:
                raise VoxCPM2BatchAdapterError(
                    f"VoxCPM2 generated no latent patches for segment {item.segment_id}."
                )
            outputs.append(torch.stack(patches, dim=0))
        return outputs

    def _decode_generated_features(self, generated: torch.Tensor) -> Any:
        latent = generated.permute(2, 0, 1).reshape(
            1,
            generated.shape[2],
            generated.shape[0] * generated.shape[1],
        )
        waveform = self.tts_model.audio_vae.decode(latent.to(torch.float32))
        waveform = waveform.detach().to("cpu").reshape(-1).numpy()
        if waveform.size == 0:
            raise VoxCPM2BatchAdapterError("VoxCPM2 decoded an empty waveform.")
        return waveform
