"""Fixtures shared by the model, training, sampling and optimizer tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles
from pbg.models.pbg.config import PBGConfig


@pytest.fixture
def tiny_config():
    """Return a factory for a one-layer PBG on the core_bin4096 vocabulary."""

    def make(**overrides) -> PBGConfig:
        values = dict(
            vocab_name="core_bin4096",
            bin_tokens=4096,
            bin_width_pm=100.0,
            max_seq_len=256,
            dim=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            intermediate_size=64,
        )
        values.update(overrides)
        return PBGConfig(**values)

    return make


@pytest.fixture
def peptide_frame():
    """Return a factory for a peptide frame in ff14SB atom order.

    Without a seed all coordinates are zero; with a seed they are small
    Gaussian displacements.
    """

    def make(sequence: str = "AA", *, seed: int | None = None, **fields) -> DataFrame:
        smiles = sequence_to_smiles(sequence)
        z = atom_numbers_from_smiles(smiles).astype(np.float32)
        if seed is None:
            xyz = np.zeros((z.size, 3), dtype=np.float32)
        else:
            xyz = np.random.default_rng(seed).normal(scale=0.3, size=(z.size, 3))
        conformer = np.concatenate([z[:, None], xyz.astype(np.float32)], axis=1)
        return DataFrame(conformer=conformer, smiles=smiles, sequence=sequence, **fields)

    return make


@pytest.fixture
def collate():
    """Return a factory that collates frame groups into a JAX training batch."""

    def make(model, groups) -> dict[str, jnp.ndarray]:
        batch = model.collate(groups, rng=np.random.default_rng(0))
        return {key: jnp.asarray(value) for key, value in batch.items()}

    return make
