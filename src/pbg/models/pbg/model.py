from __future__ import annotations

import copy
import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pbg.checkpoint import load_pickle, retire
from pbg.core.frame import DataFrame, TokenisedFrame
from pbg.core.tokeniser import (
    VocabSchema,
    load_tokeniser_default,
    upgrade_core_mask_token,
    validate_vocab,
)
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.loader import (
    augmenter_from_config,
    collate_tokenised_frames,
    prepare_frame,
)
from pbg.models.pbg.robin import (
    atom_conditioning_bias,
    init_atom_conditioner,
)
from pbg.models.pbg.transformer import (
    QwenParams,
    apply_qwen,
    atom_rope,
    init_qwen,
    qwen_params,
    qwen_vocab_size,
)

_BFLOAT16_DTYPE = jax.dtypes.bfloat16


class LossMetrics(NamedTuple):
    loss: Array
    nll: Array
    loss_tokens: Array
    self_distillation_weight_mean: Array


@dataclass(frozen=True)
class PBG:
    """Prefix-conditioned autoregressive Boltzmann generator (PBG) with explicit Qwen-style params."""

    qwen: QwenParams
    config: PBGConfig
    vocab: dict[str, object]

    @staticmethod
    def _vocab_for_config(config: PBGConfig) -> dict[str, object]:
        vocab_name = getattr(config, "vocab_name", None)
        if vocab_name is None:
            raise ValueError("config.vocab_name must name a core or joint vocabulary")
        return load_tokeniser_default(vocab_name=vocab_name)

    @staticmethod
    def _validate_vocab_embed(
        embed_size: int, vocab: dict[str, object], config: PBGConfig
    ) -> None:
        meta = vocab.get("meta", {})
        assert isinstance(meta, dict)
        next_id = int(meta["next_id"])
        if embed_size != next_id:
            raise ValueError(
                f"PBG embedding rows ({embed_size}) do not match vocab next_id ({next_id})"
            )
        validate_vocab(vocab, expected_coordinate_count=config.bin_tokens)

    @classmethod
    def load_payload(cls, payload: dict[str, object]) -> "PBG":
        """Return the model of a checkpoint payload (``params``, ``config``, ``vocab``).

        Checkpoint configs written before this release also hold the options in
        ``pbg.checkpoint.RETIRED_OPTIONS["model"]``; ``retire`` checks and drops them.
        """
        names = {field.name for field in fields(PBGConfig)}
        config = PBGConfig(**{
            key: value for key, value in retire("model", payload["config"]).items() if key in names
        })
        vocab = upgrade_core_mask_token(copy.deepcopy(payload["vocab"]))
        qwen = qwen_params(payload["params"], config)
        cls._validate_vocab_embed(qwen_vocab_size(qwen), vocab, config)
        if (config.conditioning == "atom") != ("atom_conditioner" in qwen):
            raise ValueError(
                "checkpoint atom-conditioner weights do not match config.conditioning="
                f"{config.conditioning!r}"
            )
        return cls(qwen=qwen, config=config, vocab=vocab)

    @classmethod
    def init(cls, config: PBGConfig, key: Array) -> "PBG":
        vocab = copy.deepcopy(cls._vocab_for_config(config))
        meta = vocab["meta"]
        assert isinstance(meta, dict)
        vocab_size = int(meta["next_id"])
        cls._validate_vocab_embed(vocab_size, vocab, config)
        qwen = init_qwen(key, config, vocab_size)
        if config.conditioning == "atom":
            # A separate key keeps the trunk identical to a prefix run's trunk.
            qwen["atom_conditioner"] = init_atom_conditioner(
                jax.random.fold_in(key, 1), config
            )
        return cls(qwen=qwen, config=config, vocab=vocab)

    @classmethod
    def load(cls, path: str | Path) -> "PBG":
        return cls.load_payload(load_pickle(path))

    def save(self, path: str | Path) -> None:
        with open(Path(path), "wb") as f:
            pickle.dump(
                {
                    "format_version": 2,
                    "params": jax.device_get(self.qwen),
                    "config": asdict(self.config),
                    "vocab": copy.deepcopy(self.vocab),
                },
                f,
            )

    def resize_embeddings(self, new_vocab_size: int) -> "PBG":
        """Grow embedding and optional untied output tables (new rows zeroed)."""
        embed = self.qwen["embed"]
        current = int(embed.shape[0])
        if new_vocab_size < current:
            raise ValueError(
                f"cannot shrink embedding from {current} to {new_vocab_size} rows"
            )
        if new_vocab_size > current:
            pad = jnp.zeros(
                (new_vocab_size - current, embed.shape[1]), dtype=embed.dtype
            )
            embed = jnp.concatenate([embed, pad], axis=0)
        qwen = dict(self.qwen, embed=embed)
        if "lm_head" in qwen:
            lm_head = qwen["lm_head"]
            if int(lm_head.shape[0]) != current:
                raise ValueError("lm_head rows do not match input embedding rows")
            if new_vocab_size > current:
                pad = jnp.zeros(
                    (new_vocab_size - current, lm_head.shape[1]), dtype=lm_head.dtype
                )
                lm_head = jnp.concatenate([lm_head, pad], axis=0)
            qwen["lm_head"] = lm_head
        return PBG(qwen=qwen, config=self.config, vocab=self.vocab)

    def with_vocab(self, vocab: dict[str, object]) -> "PBG":
        """Attach ``vocab`` and resize embeddings to match its ``meta.next_id``."""
        meta = vocab["meta"]
        assert isinstance(meta, dict)
        resized = self.resize_embeddings(int(meta["next_id"]))
        return PBG(
            qwen=resized.qwen, config=resized.config, vocab=copy.deepcopy(vocab)
        )

    def _decode_clamp(self) -> np.ndarray:
        # Allow coordinate tokens plus the coordinate-block terminator
        # (<EOS> legacy, <|coords_end|> joint).
        return VocabSchema.of(self.vocab).decode_bias(
            qwen_vocab_size(self.qwen), allow_terminator=True
        )

    def _atom_rope(self, atom_index: Array) -> Array:
        batch, seq_len = atom_index.shape
        seq_pos = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.float32)[None, :], (batch, seq_len)
        )
        return atom_rope(self.config, atom_index, seq_pos)

    def _level_bias(self, level_index: Array) -> Array:
        cfg = self.config
        num_levels = 1 + cfg.geom_width
        if not cfg.level_pe_enabled:
            return jnp.zeros((*level_index.shape, cfg.dim), dtype=jnp.float32)
        half = cfg.dim // 2
        levels = jnp.arange(num_levels, dtype=jnp.float32)
        freqs = 1.0 / (
            cfg.level_pe_theta ** (jnp.arange(half, dtype=jnp.float32) / max(half, 1))
        )
        table = jnp.concatenate(
            [
                jnp.sin(levels[:, None] * freqs[None, :]),
                jnp.cos(levels[:, None] * freqs[None, :]),
            ],
            axis=-1,
        )
        if table.shape[-1] < cfg.dim:
            table = jnp.concatenate(
                [
                    table,
                    jnp.zeros(
                        (num_levels, cfg.dim - table.shape[-1]), dtype=table.dtype
                    ),
                ],
                axis=-1,
            )
        return table.at[0].set(0.0)[jnp.clip(level_index, 0, num_levels - 1)]

    def collate(
        self,
        frames: list[list[DataFrame]] | list[list[TokenisedFrame]],
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        if frames and frames[0] and isinstance(frames[0][0], TokenisedFrame):
            return collate_tokenised_frames(self.config, self.vocab, frames)
        rng = np.random.default_rng() if rng is None else rng
        augmenter = augmenter_from_config(self.config)
        prepared_groups: list[list[TokenisedFrame]] = []
        for group in frames:
            prepared_group: list[TokenisedFrame] = []
            for frame in group:
                tokenised = prepare_frame(
                    frame, self.config, self.vocab, augmenter, rng=rng
                )
                if tokenised is None:
                    raise ValueError("prepare_frame returned None during collate")
                prepared_group.append(tokenised)
            prepared_groups.append(prepared_group)
        return collate_tokenised_frames(self.config, self.vocab, prepared_groups)

    def _training_masks(self, batch: dict[str, Array]) -> dict[str, Array]:
        segment_id = batch["segment_id"].astype(jnp.int32)
        attention_mask = (segment_id[:, None, :, None] >= 0) & (
            segment_id[:, None, :, None] == segment_id[:, None, None, :]
        )
        return {
            "attention_mask": attention_mask,
            "loss_mask": batch["loss_mask"].astype(jnp.float32),
        }

    def training_logits(
        self,
        batch: dict[str, Array],
        masks: dict[str, Array] | None = None,
    ) -> Array:
        """Float32 next-token logits of the training forward pass, before the decode clamp."""
        qwen = self.qwen
        activation_dtype = None
        if self.config.mixed_precision:
            # JAX 0.4.38 on LUMI does not reliably accept the jnp.bfloat16
            # scalar class as an astype dtype. Use the canonical NumPy dtype
            # while retaining the BF16 activation path.
            activation_dtype = _BFLOAT16_DTYPE
            qwen = jax.tree.map(
                lambda p: (
                    jax.lax.convert_element_type(p, activation_dtype)
                    if hasattr(p, "dtype") and jnp.issubdtype(p.dtype, jnp.floating)
                    else p
                ),
                qwen,
            )
        masks = self._training_masks(batch) if masks is None else masks
        angles = self._atom_rope(batch["atom_index"])
        bias = self._level_bias(batch["level_index"])
        if activation_dtype is not None:
            angles = jax.lax.convert_element_type(angles, activation_dtype)
            bias = jax.lax.convert_element_type(bias, activation_dtype)
        if self.config.conditioning == "atom":
            bias = atom_conditioning_bias(
                qwen["atom_conditioner"], self.config, qwen["embed"], batch, bias
            )
        logits = apply_qwen(
            qwen,
            self.config,
            batch["tokens"],
            angles=angles,
            bias=bias,
            mask=masks["attention_mask"],
            remat_blocks=self.config.remat_blocks,
        )
        return logits.astype(jnp.float32)

    def target_log_probs(
        self,
        batch: dict[str, Array],
        masks: dict[str, Array] | None = None,
    ) -> Array:
        logits = self.training_logits(batch, masks)
        decode_bias = self._decode_clamp()
        lp = jax.nn.log_softmax(logits + jnp.asarray(decode_bias), axis=-1)
        chosen = jnp.take_along_axis(
            lp, batch["targets"].astype(jnp.int32)[..., None], axis=-1
        ).squeeze(-1)
        return chosen

    def loss_metrics(self, batch: dict[str, Array]) -> LossMetrics:
        masks = self._training_masks(batch)
        chosen_lp = self.target_log_probs(batch, masks=masks)
        mask = masks["loss_mask"]
        token_nll = -jnp.where(mask > 0, chosen_lp.astype(jnp.float32), 0.0)
        self_weight = batch.get("sample_weight", jnp.ones_like(mask)).astype(
            jnp.float32
        )
        active = mask.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(active), 1.0)
        if self.config.loss_reduction == "sample":
            # ROBIN: the mean NLL of each molecule, then the mean over the
            # molecules, so every molecule has the same weight. A molecule
            # never spans rows, so (row, segment) identifies it. Only
            # row-major arrays reach the trainer's devices (put_batch).
            rows, length = mask.shape
            n_slots = rows * length
            owner = jnp.where(
                active > 0,
                jnp.arange(rows, dtype=jnp.int32)[:, None] * length
                + batch["segment_id"].astype(jnp.int32),
                n_slots,
            )

            def per_sample(values: Array) -> Array:
                return jax.ops.segment_sum(
                    values.ravel(), owner.ravel(), num_segments=n_slots + 1
                )[:n_slots]

            counts = per_sample(active)
            present = (counts > 0).astype(jnp.float32)
            molecules = jnp.maximum(jnp.sum(present), 1.0)
            loss = jnp.sum(
                per_sample(token_nll * active * self_weight) / jnp.maximum(counts, 1.0)
            ) / molecules
            nll = jnp.sum(per_sample(token_nll * active) / jnp.maximum(counts, 1.0)) / molecules
        else:
            loss = jnp.sum(token_nll * active * self_weight) / denominator
            nll = jnp.sum(token_nll * active) / denominator
        return LossMetrics(
            loss=loss,
            nll=nll,
            loss_tokens=jnp.sum(active),
            self_distillation_weight_mean=jnp.sum(self_weight * active) / denominator,
        )

    def loss(self, batch: dict[str, Array]) -> Array:
        return self.loss_metrics(batch).loss

    def score(self, frames: list[DataFrame], key: Array | None = None) -> Array:
        if not frames:
            return jnp.zeros((0,), dtype=jnp.float32)
        if any(f.conformer is None for f in frames):
            raise ValueError("score requires conformers")
        seed = int(jax.random.bits(key)) % (2**31) if key is not None else 0
        batch_np = self.collate([[f] for f in frames], rng=np.random.default_rng(seed))
        batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
        chosen_lp = self.target_log_probs(batch)
        terminator_id = VocabSchema.of(self.vocab).terminator_id
        active = batch["loss_mask"].astype(jnp.float32) * (
            batch["targets"] != terminator_id
        ).astype(jnp.float32)
        return (jnp.where(active > 0, chosen_lp, 0.0) * active).sum(axis=1)
