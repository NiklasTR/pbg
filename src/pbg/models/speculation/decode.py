"""Batched Leviathan AR draft–verify for large-batch junior speculative decode."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class DecodeResult(NamedTuple):
    tokens: Array
    target_log_probs: Array
    proposed: int
    accepted: int
    forwards: int
    forward_tokens: int = 0
    proposal1_attempted: int = 0
    proposal1_accepted: int = 0
    proposal2_eligible: int = 0
    proposal2_accepted: int = 0
    depth_attempted: Array | None = None
    depth_accepted: Array | None = None


def _compact_distribution(
    logits: Array, legal_ids: Array | None, temperature: float
) -> Array:
    selected = logits if legal_ids is None else logits[..., legal_ids]
    return jax.nn.softmax(selected.astype(jnp.float32) / temperature)


def _sample_compact(key: Array, probabilities: Array) -> Array:
    return jax.random.categorical(
        key, jnp.log(jnp.maximum(probabilities, 1e-30)), axis=-1
    ).astype(jnp.int32)


def _propose_compact(
    key: Array, probabilities: Array, proposal_policy: str
) -> tuple[Array, Array]:
    if proposal_policy == "sample":
        return _sample_compact(key, probabilities), probabilities
    if proposal_policy == "argmax":
        tokens = jnp.argmax(probabilities, axis=-1).astype(jnp.int32)
        proposal_law = jax.nn.one_hot(
            tokens, probabilities.shape[-1], dtype=probabilities.dtype
        )
        return tokens, proposal_law
    raise ValueError("proposal_policy must be 'sample' or 'argmax'")


def decode_compiled_ar(
    key: Array,
    forward,
    prompt_tokens: Array,
    prompt_atom_index: Array,
    prompt_level_index: Array,
    coordinate_atom_index: Array,
    coordinate_level_index: Array,
    legal_token_ids: Array,
    *,
    temperature: float = 1.0,
    compact_logits: bool = False,
) -> DecodeResult:
    """Fully staged batch-one cached AR decode.

    The caller should JIT this function with ``forward`` closed over. The entire
    token loop, cache state, sampling, and output writes then remain on device.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    horizon = coordinate_atom_index.shape[0]
    prompt_len = prompt_tokens.shape[0]
    if prompt_len == 0:
        raise ValueError("compiled AR requires a non-empty prompt")

    logits, caches = forward(
        prompt_tokens,
        jnp.zeros((prompt_len,), dtype=bool),
        prompt_atom_index,
        prompt_level_index,
        jnp.arange(prompt_len, dtype=jnp.int32),
        None,
        jnp.int32(0),
    )
    tokens = jnp.zeros((horizon,), dtype=jnp.int32)
    log_probs = jnp.zeros((horizon,), dtype=jnp.float32)

    def body(state):
        index, key, next_logits, caches, tokens, log_probs = state
        key, sample_key = jax.random.split(key)
        probabilities = _compact_distribution(
            next_logits,
            None if compact_logits else legal_token_ids,
            temperature,
        )
        compact_token = _sample_compact(sample_key, probabilities)
        token = legal_token_ids[compact_token]
        tokens = tokens.at[index].set(token)
        log_probs = log_probs.at[index].set(
            jnp.log(jnp.maximum(probabilities[compact_token], 1e-30))
        )

        def extend(args):
            caches, token = args
            position = prompt_len + index
            next_result, next_caches = forward(
                token[None],
                jnp.zeros((1,), dtype=bool),
                jax.lax.dynamic_slice(coordinate_atom_index, (index,), (1,)),
                jax.lax.dynamic_slice(coordinate_level_index, (index,), (1,)),
                jnp.asarray([position], dtype=jnp.int32),
                caches,
                jnp.asarray(position, dtype=jnp.int32),
            )
            return next_result[0], next_caches

        next_logits, caches = jax.lax.cond(
            index + 1 < horizon,
            extend,
            lambda args: (next_logits, args[0]),
            (caches, token),
        )
        return index + 1, key, next_logits, caches, tokens, log_probs

    initial = (
        jnp.int32(0),
        key,
        logits[-1],
        caches,
        tokens,
        log_probs,
    )
    index, _, _, _, tokens, log_probs = jax.lax.while_loop(
        lambda state: state[0] < horizon, body, initial
    )
    return DecodeResult(
        tokens,
        log_probs,
        jnp.int32(0),
        jnp.int32(0),
        index,
        prompt_len + jnp.maximum(index - 1, 0),
    )



def decode_compiled_batched_junior_ar(
    keys: Array,
    draft_forward,
    target_forward,
    prompt_tokens: Array,
    prompt_atom_index: Array,
    prompt_level_index: Array,
    coordinate_atom_index: Array,
    coordinate_level_index: Array,
    legal_token_ids: Array,
    *,
    num_proposals: int,
    temperature: float = 1.0,
    compact_logits: bool = True,
    proposal_policy: str = "sample",
) -> DecodeResult:
    """Batch-native Leviathan AR speculative decode.

    Each round teacher-forces the pending token, drafts ``K`` tokens with the
    junior model (``K+1`` width-1 forwards), then verifies them with one packed
    target forward of width ``K+1``. Accepted KV is kept by advancing
    ``cache_index``; rejected suffixes stay past that index and are overwritten.
    A bonus target token is emitted on a full accept when horizon remains.
    """
    if num_proposals <= 0:
        raise ValueError("num_proposals must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if proposal_policy not in {"sample", "argmax"}:
        raise ValueError("proposal_policy must be 'sample' or 'argmax'")
    if keys.ndim < 1:
        raise ValueError("keys must have a leading batch dimension")
    horizon = coordinate_atom_index.shape[0]
    prompt_len = prompt_tokens.shape[0]
    if prompt_len < 2:
        raise ValueError("junior speculative decode requires at least two prompt tokens")
    if horizon <= 0:
        raise ValueError("junior speculative decode requires a positive horizon")

    batch = keys.shape[0]
    k = int(num_proposals)
    prefill_len = prompt_len - 1
    compact_ids = None if compact_logits else legal_token_ids

    def broadcast(value):
        return jnp.broadcast_to(value, (batch, *value.shape))

    def choose(live, new, old):
        mask = live.reshape((batch,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    prefill_mask = jnp.zeros((batch, prefill_len), dtype=bool)
    prefill_positions = broadcast(jnp.arange(prefill_len, dtype=jnp.int32))
    prefill_index = jnp.zeros((batch,), dtype=jnp.int32)
    _, draft_caches = draft_forward(
        broadcast(prompt_tokens[:prefill_len]),
        prefill_mask,
        broadcast(prompt_atom_index[:prefill_len]),
        broadcast(prompt_level_index[:prefill_len]),
        prefill_positions,
        None,
        prefill_index,
    )
    _, target_caches = target_forward(
        broadcast(prompt_tokens[:prefill_len]),
        prefill_mask,
        broadcast(prompt_atom_index[:prefill_len]),
        broadcast(prompt_level_index[:prefill_len]),
        prefill_positions,
        None,
        prefill_index,
    )

    padded_atoms = jnp.pad(coordinate_atom_index, (0, k), mode="edge")
    padded_levels = jnp.pad(coordinate_level_index, (0, k), mode="edge")
    tokens = jnp.zeros((batch, horizon), dtype=jnp.int32)
    log_probs = jnp.zeros((batch, horizon), dtype=jnp.float32)
    batch_ids = jnp.arange(batch, dtype=jnp.int32)
    rows = jnp.arange(k, dtype=jnp.int32)
    slice_draft_metadata = jax.vmap(
        lambda values, start: jax.lax.dynamic_slice(values, (start,), (k,)),
        in_axes=(None, 0),
    )
    split_pair = jax.vmap(lambda key: jax.random.split(key))
    split_four = jax.vmap(lambda key: jax.random.split(key, 4))
    sample_rows = jax.vmap(_sample_compact)
    propose_rows = jax.vmap(
        lambda key, probabilities: _propose_compact(
            key, probabilities, proposal_policy
        )
    )
    uniform_rows = jax.vmap(lambda key: jax.random.uniform(key, (k,)))

    def body(state):
        (
            output_count,
            keys,
            draft_caches,
            target_caches,
            cache_index,
            pending,
            pending_atom,
            pending_level,
            pending_position,
            tokens,
            log_probs,
            proposed,
            accepted,
            proposal1_attempted,
            proposal1_accepted,
            proposal2_eligible,
            proposal2_accepted,
            depth_attempted,
            depth_accepted,
            rounds,
        ) = state
        live = output_count < horizon
        remaining = horizon - output_count
        active = jnp.where(live, jnp.minimum(jnp.int32(k), remaining), jnp.int32(0))
        safe_output = jnp.where(live, output_count, 0)
        safe_cache_index = jnp.where(live, cache_index, 0)
        draft_atoms = slice_draft_metadata(padded_atoms, safe_output)
        draft_levels = slice_draft_metadata(padded_levels, safe_output)

        round_keys = split_four(keys)
        keys = round_keys[:, 0]
        accept_keys = round_keys[:, 1]
        correction_keys = round_keys[:, 2]
        bonus_keys = round_keys[:, 3]

        draft_tok = jnp.zeros((batch, k), dtype=jnp.int32)
        draft_compact = jnp.zeros((batch, k), dtype=jnp.int32)
        draft_q = jnp.zeros((batch, k, legal_token_ids.shape[0]), dtype=jnp.float32)

        def draft_step(step, carry):
            (
                keys,
                draft_caches,
                feed,
                feed_atom,
                feed_level,
                feed_position,
                draft_tok,
                draft_compact,
                draft_q,
            ) = carry
            logits, draft_caches = draft_forward(
                feed[:, None],
                jnp.zeros((batch, 1), dtype=bool),
                feed_atom[:, None],
                feed_level[:, None],
                feed_position[:, None],
                draft_caches,
                safe_cache_index + step,
            )
            probabilities = _compact_distribution(
                logits[:, 0], compact_ids, temperature
            )
            pair = split_pair(keys)
            keys = pair[:, 0]
            sample_keys = pair[:, 1]
            compact, proposal_law = propose_rows(sample_keys, probabilities)
            token = legal_token_ids[compact]
            write = live & (step < active)
            draft_tok = draft_tok.at[:, step].set(
                jnp.where(write, token, draft_tok[:, step])
            )
            draft_compact = draft_compact.at[:, step].set(
                jnp.where(write, compact, draft_compact[:, step])
            )
            draft_q = draft_q.at[:, step].set(
                jnp.where(write[:, None], proposal_law, draft_q[:, step])
            )
            return (
                keys,
                draft_caches,
                token,
                draft_atoms[:, step],
                draft_levels[:, step],
                feed_position + jnp.int32(1),
                draft_tok,
                draft_compact,
                draft_q,
            )

        (
            keys,
            draft_caches,
            last_draft,
            last_atom,
            last_level,
            last_position,
            draft_tok,
            draft_compact,
            draft_q,
        ) = jax.lax.fori_loop(
            0,
            k,
            draft_step,
            (
                keys,
                draft_caches,
                pending,
                pending_atom,
                pending_level,
                pending_position,
                draft_tok,
                draft_compact,
                draft_q,
            ),
        )
        _, draft_caches = draft_forward(
            last_draft[:, None],
            jnp.zeros((batch, 1), dtype=bool),
            last_atom[:, None],
            last_level[:, None],
            last_position[:, None],
            draft_caches,
            safe_cache_index + k,
        )

        verify_tokens = jnp.concatenate((pending[:, None], draft_tok), axis=1)
        verify_atoms = jnp.concatenate((pending_atom[:, None], draft_atoms), axis=1)
        verify_levels = jnp.concatenate((pending_level[:, None], draft_levels), axis=1)
        verify_positions = (
            pending_position[:, None] + jnp.arange(k + 1, dtype=jnp.int32)[None]
        )
        verify_logits, target_caches = target_forward(
            verify_tokens,
            jnp.zeros((batch, k + 1), dtype=bool),
            verify_atoms,
            verify_levels,
            verify_positions,
            target_caches,
            safe_cache_index,
        )
        target_p = _compact_distribution(
            verify_logits[:, :k], compact_ids, temperature
        )
        bonus_p = _compact_distribution(
            verify_logits[batch_ids, active], compact_ids, temperature
        )

        eligible = rows[None] < active[:, None]
        p_selected = target_p[batch_ids[:, None], rows[None], draft_compact]
        q_selected = draft_q[batch_ids[:, None], rows[None], draft_compact]
        accept_probability = jnp.minimum(
            1.0, p_selected / jnp.maximum(q_selected, 1e-30)
        )
        accept_draws = uniform_rows(accept_keys) < accept_probability
        accepted_prefix = jnp.cumprod(
            (accept_draws | ~eligible).astype(jnp.int32), axis=1
        )
        accepted_mask = eligible & accepted_prefix.astype(bool)
        accepted_count = jnp.sum(accepted_mask, axis=1, dtype=jnp.int32)
        rejected = accepted_count < active
        rejection_index = jnp.minimum(accepted_count, jnp.int32(k - 1))
        has_proposal1 = active >= 1
        proposal1_was_accepted = accepted_count >= 1
        proposal2_was_eligible = (active >= 2) & proposal1_was_accepted
        proposal2_was_accepted = accepted_count >= 2
        depth_was_attempted = eligible & (accepted_count[:, None] >= rows[None])
        depth_was_accepted = accepted_count[:, None] > rows[None]

        rejection_target = target_p[batch_ids, rejection_index]
        rejection_proposal = draft_q[batch_ids, rejection_index]
        residual = jnp.maximum(rejection_target - rejection_proposal, 0.0)
        residual_mass = jnp.sum(residual, axis=-1, keepdims=True)
        residual = jnp.where(
            residual_mass > 0, residual / residual_mass, rejection_target
        )
        correction_compact = sample_rows(correction_keys, residual)
        bonus_compact = sample_rows(bonus_keys, bonus_p)
        can_bonus = live & (~rejected) & (remaining > active)
        extra = rejected | can_bonus
        emitted_count = jnp.where(
            live,
            jnp.minimum(accepted_count + extra.astype(jnp.int32), remaining),
            jnp.int32(0),
        )

        def write_emitted(index, carry):
            tokens, log_probs = carry
            write = live & (index < emitted_count)
            output_index = jnp.minimum(output_count + index, horizon - 1)
            is_extra = index == accepted_count
            compact = jnp.where(
                is_extra & rejected,
                correction_compact,
                jnp.where(
                    is_extra,
                    bonus_compact,
                    draft_compact[batch_ids, jnp.minimum(index, jnp.int32(k - 1))],
                ),
            )
            token = legal_token_ids[compact]
            p_row = jnp.where(
                (is_extra & (~rejected))[:, None],
                bonus_p,
                target_p[batch_ids, jnp.minimum(index, jnp.int32(k - 1))],
            )
            old_token = tokens[batch_ids, output_index]
            old_log_prob = log_probs[batch_ids, output_index]
            tokens = tokens.at[batch_ids, output_index].set(
                jnp.where(write, token, old_token)
            )
            value = jnp.log(jnp.maximum(p_row[batch_ids, compact], 1e-30))
            log_probs = log_probs.at[batch_ids, output_index].set(
                jnp.where(write, value, old_log_prob)
            )
            return tokens, log_probs

        tokens, log_probs = jax.lax.fori_loop(
            0, k + 1, write_emitted, (tokens, log_probs)
        )
        new_output_count = output_count + emitted_count
        pending_index = jnp.minimum(
            jnp.maximum(new_output_count - 1, 0), horizon - 1
        )
        new_pending = tokens[batch_ids, pending_index]
        new_pending_atom = padded_atoms[pending_index]
        new_pending_level = padded_levels[pending_index]
        new_pending_position = prompt_len + pending_index
        committed = jnp.int32(1) + accepted_count
        return (
            jnp.where(live, new_output_count, output_count),
            keys,
            draft_caches,
            target_caches,
            jnp.where(live, cache_index + committed, cache_index),
            choose(live, new_pending, pending),
            choose(live, new_pending_atom, pending_atom),
            choose(live, new_pending_level, pending_level),
            choose(live, new_pending_position, pending_position),
            tokens,
            log_probs,
            proposed + jnp.where(live, active, 0),
            accepted + jnp.where(live, accepted_count, 0),
            proposal1_attempted + jnp.where(live, has_proposal1, 0),
            proposal1_accepted + jnp.where(live, proposal1_was_accepted, 0),
            proposal2_eligible + jnp.where(live, proposal2_was_eligible, 0),
            proposal2_accepted + jnp.where(live, proposal2_was_accepted, 0),
            depth_attempted + jnp.where(live[:, None], depth_was_attempted, 0),
            depth_accepted + jnp.where(live[:, None], depth_was_accepted, 0),
            rounds + live.astype(jnp.int32),
        )

    zeros = jnp.zeros((batch,), dtype=jnp.int32)
    initial = (
        zeros,
        keys,
        draft_caches,
        target_caches,
        jnp.full((batch,), prefill_len, dtype=jnp.int32),
        jnp.full((batch,), prompt_tokens[-1], dtype=jnp.int32),
        jnp.full((batch,), prompt_atom_index[-1], dtype=jnp.int32),
        jnp.full((batch,), prompt_level_index[-1], dtype=jnp.int32),
        jnp.full((batch,), prompt_len - 1, dtype=jnp.int32),
        tokens,
        log_probs,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.zeros((batch, k), dtype=jnp.int32),
        jnp.zeros((batch, k), dtype=jnp.int32),
        zeros,
    )
    result = jax.lax.while_loop(
        lambda state: jnp.any(state[0] < horizon), body, initial
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        tokens,
        log_probs,
        proposed,
        accepted,
        proposal1_attempted,
        proposal1_accepted,
        proposal2_eligible,
        proposal2_accepted,
        depth_attempted,
        depth_accepted,
        rounds,
    ) = result
    return DecodeResult(
        tokens,
        log_probs,
        proposed,
        accepted,
        2 + (k + 2) * rounds,
        2 * prefill_len + (2 * (k + 1)) * rounds,
        proposal1_attempted,
        proposal1_accepted,
        proposal2_eligible,
        proposal2_accepted,
        depth_attempted,
        depth_accepted,
    )


