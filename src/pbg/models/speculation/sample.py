"""Sample conformers with junior AR speculative decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pbg.core.tokeniser import VocabSchema, undo_tokeniser_coordinate
from pbg.models.pbg.model import PBG
from pbg.models.pbg.sample import SampleBatch
from pbg.models.pbg.system import MolecularSystem, prompt_arrays

from .decode import DecodeResult, decode_compiled_ar, decode_compiled_batched_junior_ar
from .inference import cast_base_inference, make_base_forward, make_batched_base_forward


@dataclass(frozen=True, slots=True)
class SamplingMetrics:
    method: str
    batch_size: int
    samples: int
    coordinate_tokens: int
    compile_seconds: float
    generation_seconds: float
    proposals: int
    accepted: int
    proposal1_attempted: int
    proposal1_accepted: int
    proposal2_eligible: int
    proposal2_accepted: int
    forwards: int
    forward_tokens: int
    depth_attempted: tuple[int, ...] = ()
    depth_accepted: tuple[int, ...] = ()

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.generation_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.coordinate_tokens / self.generation_seconds

    @property
    def generated_tokens_per_forward(self) -> float:
        return self.coordinate_tokens / self.forwards if self.forwards else 0.0


def _assert_compatible_vocabs(draft: PBG, target: PBG) -> None:
    draft_ids = tuple(int(value) for value in VocabSchema.of(draft.vocab).coord_ids)
    target_ids = tuple(int(value) for value in VocabSchema.of(target.vocab).coord_ids)
    if draft_ids != target_ids:
        raise ValueError(
            "draft and target must share the same coordinate vocabulary "
            f"(draft {len(draft_ids)} ids, target {len(target_ids)} ids)"
        )
    if draft.config.bin_tokens != target.config.bin_tokens:
        raise ValueError("draft and target bin_tokens must match")


def _reconstruct(
    model: PBG, token_ids: np.ndarray, coord_len: int, z: np.ndarray
) -> np.ndarray:
    schema = VocabSchema.of(model.vocab)
    local_ids = token_ids - schema.coord_base_id
    n_batch = token_ids.shape[0]
    depth = coord_len // int(z.size)
    geometry = local_ids.reshape(n_batch, z.size, depth)
    xyz = undo_tokeniser_coordinate(
        geometry,
        "bin",
        num_bins=model.config.bin_tokens,
        min_val=model.config.min_val,
        max_val=model.config.max_val,
        decode_method="center",
    )
    return np.concatenate(
        [np.broadcast_to(z[None, :, None], (n_batch, z.size, 1)), xyz], axis=2
    )


def sample_junior_conformers(
    draft: PBG,
    target: PBG,
    system: MolecularSystem,
    key: Array,
    *,
    num_samples: int,
    batch_size: int,
    num_proposals: int,
    temperature: float = 1.0,
    proposal_policy: str = "sample",
) -> tuple[SampleBatch, SamplingMetrics]:
    """Sample target-distributed conformers via junior AR draft–verify."""
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if num_proposals <= 0:
        raise ValueError("num_proposals must be positive")
    _assert_compatible_vocabs(draft, target)
    draft = cast_base_inference(draft, mixed_precision=True)
    target = cast_base_inference(target, mixed_precision=True)
    k = int(num_proposals)
    prompt, atom_index, level_index, prompt_len, coord_len, z = prompt_arrays(
        target, system, 1
    )
    legal_ids = jnp.asarray(VocabSchema.of(target.vocab).coord_ids)
    common = dict(
        prompt_tokens=jnp.asarray(prompt[0]),
        prompt_atom_index=jnp.asarray(atom_index[0, :prompt_len]),
        prompt_level_index=jnp.asarray(level_index[0, :prompt_len]),
        coordinate_atom_index=jnp.asarray(
            atom_index[0, prompt_len : prompt_len + coord_len]
        ),
        coordinate_level_index=jnp.asarray(
            level_index[0, prompt_len : prompt_len + coord_len]
        ),
        legal_token_ids=legal_ids,
        temperature=temperature,
        num_proposals=k,
        compact_logits=True,
        proposal_policy=proposal_policy,
    )
    max_cache_len = prompt_len + coord_len + k
    draft_forward = make_batched_base_forward(
        draft,
        extension_len=1,
        output_token_ids=legal_ids,
        max_cache_len=max_cache_len,
    )
    target_forward = make_batched_base_forward(
        target,
        extension_len=k + 1,
        output_token_ids=legal_ids,
        max_cache_len=max_cache_len,
    )
    decode = jax.jit(
        lambda sample_keys: decode_compiled_batched_junior_ar(
            sample_keys,
            draft_forward,
            target_forward,
            **common,
        )
    )
    return _sample_batches(
        target,
        decode,
        key,
        method=f"junior_k{k}",
        num_samples=num_samples,
        batch_size=batch_size,
        coord_len=coord_len,
        z=z,
    )


def sample_compiled_ar_conformers(
    base,
    system: MolecularSystem,
    key: Array,
    *,
    num_samples: int,
    batch_size: int,
    temperature: float = 1.0,
) -> tuple[SampleBatch, SamplingMetrics]:
    """Sample conformers with the same compiled native AR benchmark path."""
    base = cast_base_inference(base, mixed_precision=True)
    prompt, atom_index, level_index, prompt_len, coord_len, z = prompt_arrays(
        base, system, 1
    )
    common = dict(
        prompt_tokens=jnp.asarray(prompt[0]),
        prompt_atom_index=jnp.asarray(atom_index[0, :prompt_len]),
        prompt_level_index=jnp.asarray(level_index[0, :prompt_len]),
        coordinate_atom_index=jnp.asarray(
            atom_index[0, prompt_len : prompt_len + coord_len]
        ),
        coordinate_level_index=jnp.asarray(
            level_index[0, prompt_len : prompt_len + coord_len]
        ),
        legal_token_ids=jnp.asarray(VocabSchema.of(base.vocab).coord_ids),
        temperature=temperature,
    )
    forward = make_base_forward(
        base,
        output_token_ids=common["legal_token_ids"],
        max_cache_len=prompt_len + coord_len,
    )
    decode = jax.jit(
        jax.vmap(
            lambda sample_key: decode_compiled_ar(
                sample_key,
                forward,
                compact_logits=True,
                **common,
            )
        )
    )
    return _sample_batches(
        base,
        decode,
        key,
        method="ar",
        num_samples=num_samples,
        batch_size=batch_size,
        coord_len=coord_len,
        z=z,
    )


def _sample_batches(
    base,
    decode,
    key: Array,
    *,
    method: str,
    num_samples: int,
    batch_size: int,
    coord_len: int,
    z: np.ndarray,
) -> tuple[SampleBatch, SamplingMetrics]:
    batches = (num_samples + batch_size - 1) // batch_size
    padded_samples = batches * batch_size
    keys = jax.random.split(key, padded_samples).reshape(
        (batches, batch_size, *key.shape)
    )
    decode_many = jax.jit(lambda batch_keys: jax.lax.map(decode, batch_keys))
    started = time.perf_counter()
    compiled = decode_many.lower(keys).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = compiled(keys)
    jax.block_until_ready(result.tokens)
    generation_seconds = time.perf_counter() - started
    token_ids = np.asarray(result.tokens, dtype=np.int32).reshape(
        padded_samples, coord_len
    )[:num_samples]
    target_log_probs = np.asarray(result.target_log_probs, dtype=np.float32).reshape(
        padded_samples, coord_len
    )[:num_samples]
    totals = {}
    for name in DecodeResult._fields[2:10]:
        value = np.asarray(jax.device_get(getattr(result, name))).reshape(
            padded_samples
        )
        totals[name] = int(value[:num_samples].sum())
    if result.depth_attempted is None:
        depth_attempted = np.zeros((num_samples, 0), dtype=np.int32)
        depth_accepted = np.zeros((num_samples, 0), dtype=np.int32)
    else:
        depth_attempted = np.asarray(jax.device_get(result.depth_attempted)).reshape(
            padded_samples, -1
        )[:num_samples]
        depth_accepted = np.asarray(jax.device_get(result.depth_accepted)).reshape(
            padded_samples, -1
        )[:num_samples]
    samples = _reconstruct(base, token_ids, coord_len, z).astype(np.float64)
    schema = VocabSchema.of(base.vocab)
    behavior_lp = np.concatenate(
        (target_log_probs, np.zeros((num_samples, 1), dtype=np.float32)), axis=1
    )
    behavior_ids = np.concatenate(
        (
            token_ids,
            np.full((num_samples, 1), schema.terminator_id, dtype=np.int32),
        ),
        axis=1,
    )
    batch = SampleBatch(
        samples=samples,
        logq=np.sum(target_log_probs, axis=1, dtype=np.float64),
        behavior_token_log_probs=behavior_lp,
        behavior_token_ids=behavior_ids,
    )
    metrics = SamplingMetrics(
        method=method,
        batch_size=batch_size,
        samples=num_samples,
        coordinate_tokens=num_samples * coord_len,
        compile_seconds=compile_seconds,
        generation_seconds=generation_seconds,
        proposals=totals["proposed"],
        accepted=totals["accepted"],
        proposal1_attempted=totals["proposal1_attempted"],
        proposal1_accepted=totals["proposal1_accepted"],
        proposal2_eligible=totals["proposal2_eligible"],
        proposal2_accepted=totals["proposal2_accepted"],
        forwards=totals["forwards"],
        forward_tokens=totals["forward_tokens"],
        depth_attempted=tuple(int(value) for value in depth_attempted.sum(axis=0)),
        depth_accepted=tuple(int(value) for value in depth_accepted.sum(axis=0)),
    )
    return batch, metrics
