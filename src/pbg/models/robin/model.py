"""Load converted ROBIN checkpoints and run the historical CGPT decoder."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pbg.checkpoint import load_pickle
from pbg.core.frame import DataFrame
from pbg.core.tokeniser import undo_tokeniser_coordinate

from .cgpt import apply_model
from .conditioning import encode_sequences
from .config import RobinConfig


@dataclass(frozen=True)
class Robin:
    """ROBIN inference interface; parameters use the converted JAX layout."""

    params: dict
    config: RobinConfig

    @classmethod
    def load(cls, path: str | Path) -> 'Robin':
        """Load a trusted converted JAX checkpoint, not a PyTorch checkpoint."""
        payload = load_pickle(path)
        config = RobinConfig.from_checkpoint(payload['config'])
        params = jax.tree.map(jnp.asarray, payload['params'])
        required = {'token_embed', 'pos_embed', 'coord_dim_embed', 'atom_pos_embed',
                    'cond_embed', 'global_cond_attn', 'blocks', 'final_norm', 'output_proj_w'}
        if required - params.keys():
            raise ValueError(f'Missing CGPT parameters: {sorted(required - params.keys())}')
        if len(params['blocks']) != config.num_layers:
            raise ValueError('Checkpoint layer count does not match its configuration')
        if params['token_embed'].shape[1] != config.dim:
            raise ValueError('Checkpoint embedding dimension does not match its configuration')
        return cls(params, config)


    def collate(self, frames: list[DataFrame]) -> dict:
        """Prepare atom-conditioned inputs for incomplete peptide frames."""
        if not frames or any(frame.sequence is None or frame.conformer is not None for frame in frames):
            raise ValueError('ROBIN collate requires incomplete peptide frames with sequences')
        encodings, mask = encode_sequences([frame.sequence for frame in frames], self.config.max_num_atoms)
        lengths = np.asarray([frame.n_atoms * 3 for frame in frames], dtype=np.int32)
        if not np.array_equal(mask.sum(axis=1).astype(np.int32) * 3, lengths):
            raise ValueError('ROBIN conditioning atom count differs from the peptide topology')
        return {**encodings, 'mask': mask, 'lengths': lengths}

    def predict(self, batch, key, max_seq_len: int, temperature: float = 1.0, *, params=None):
        """Return coordinate tokens and log likelihoods with cached AR decoding.

        Wrap in jax.jit with static max_seq_len and temperature. Preserve the
        original zero start token, float32 caches, and untempered logq at T != 1.
        Sampling benchmarks use T=1 so these are the sampling log likelihoods.
        """
        if temperature < 0:
            raise ValueError('temperature must be nonnegative')
        params = self.params if params is None else params
        cfg = self.config
        if not 0 < max_seq_len <= cfg.max_num_atoms * 3:
            raise ValueError('Invalid coordinate sequence length')
        if params['pos_embed'].shape[0] < max_seq_len or params['atom_pos_embed'].shape[0] < (max_seq_len + 2) // 3:
            raise ValueError('Peptide exceeds the checkpoint positional embedding capacity')
        encodings = {name: batch[name] for name in ('atom_type', 'aa_type', 'aa_pos', 'atom_pos', 'seq_len')}
        batch_size = batch['atom_type'].shape[0]
        cache_shape = (cfg.num_layers, batch_size, cfg.max_num_atoms * 3 + 1, cfg.num_heads, cfg.dim // cfg.num_heads)
        initial = (
            jnp.zeros((batch_size, max_seq_len + 1), dtype=jnp.int32), key,
            jnp.zeros(cache_shape), jnp.zeros(cache_shape),
            jnp.zeros((batch_size, max_seq_len), dtype=jnp.float32),
        )

        def step(index, state):
            ids, rng, ck, cv, lp = state
            rng, sample_key = jax.random.split(rng)
            token = jax.lax.dynamic_slice(ids, (0, index), (batch_size, 1))
            logits, caches = apply_model(
                params, {'dim': cfg.dim, 'num_heads': cfg.num_heads}, token, encodings,
                mask=batch['mask'], kv_cache=[(ck[layer], cv[layer]) for layer in range(cfg.num_layers)],
                cache_index=index,
            )
            logits = logits[:, 0]
            next_token = (jax.random.categorical(sample_key, logits / temperature)
                          if temperature > 0 else jnp.argmax(logits, axis=-1))
            chosen = jax.nn.log_softmax(logits)[jnp.arange(batch_size), next_token]
            return (ids.at[:, index + 1].set(next_token), rng,
                    jnp.stack([cache[0] for cache in caches]), jnp.stack([cache[1] for cache in caches]),
                    lp.at[:, index].set(chosen))

        ids, _, _, _, log_probs = jax.lax.fori_loop(0, max_seq_len, step, initial)
        valid = jnp.arange(max_seq_len) < batch['lengths'][:, None]
        return ids[:, 1:], jnp.sum(jnp.where(valid, log_probs, 0), axis=1)

    def decollate(self, tokens, lengths, key=None) -> list[np.ndarray]:
        """Reconstruct coordinates with the shared coordinate dequantizer."""
        lengths = np.asarray(lengths)
        if not lengths.size:
            return []
        cfg = self.config
        n_atoms = int(lengths.max()) // 3
        flat = np.asarray(jax.device_get(tokens[:, :n_atoms * 3]))
        if cfg.token_mode == 'combined':
            flat = flat % cfg.num_bins
        positions = np.asarray(undo_tokeniser_coordinate(
            flat.reshape(tokens.shape[0], n_atoms, 3), cfg.coordinate_method,
            num_bins=cfg.num_bins, min_val=cfg.min_val, max_val=cfg.max_val,
            decode_method='uniform' if cfg.undiscretize_method == 'uniform_noise' else cfg.undiscretize_method,
            key=key,
        ))
        return [positions[index, :int(length) // 3] for index, length in enumerate(lengths)]
