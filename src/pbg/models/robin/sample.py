"""Production ROBIN ancestral sampling with the historical JIT predict path."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pbg.core.frame import DataFrame
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.models.pbg.sample import SampleBatch
from pbg.models.pbg.system import MolecularSystem

from .model import Robin


@partial(jax.jit, static_argnames=('config', 'max_seq_len', 'temperature'))
def predict(params, config, batch, key, max_seq_len, temperature):
    return Robin(params, config).predict(batch, key, max_seq_len, temperature)


def sample_robin_conformers(
    model: Robin, system: MolecularSystem, key, *, num_samples: int,
    batch_size: int = 512, temperature: float = 1.0,
) -> SampleBatch:
    """Return zxyz conformers and original model logq, without energy or export.

    Match the old ARBG sampler: tune the runtime atom padding to the peptide,
    collate a template batch, use cached predict, and reconstruct on the host.
    No reflection or stereochemistry filter is applied here.
    """
    if num_samples < 1 or batch_size < 1:
        raise ValueError('num_samples and batch_size must be positive')
    if not system.sequence:
        raise ValueError('ROBIN requires a peptide sequence')
    template = DataFrame.from_sequence(system.sequence)
    n_atoms = template.n_atoms
    model = replace(model, config=replace(model.config, max_num_atoms=n_atoms))
    atoms = atom_numbers_from_smiles(template.smiles)
    atoms = np.asarray(atoms, dtype=np.float64)
    if atoms.size != n_atoms:
        raise ValueError('ROBIN atom ordering does not match the input frame')
    frames = [replace(template, frame_id=index) for index in range(min(batch_size, num_samples))]
    batch = model.collate(frames)
    lengths = batch['lengths']
    jax_batch = jax.tree.map(jnp.asarray, batch)
    positions, logq = [], []
    for start in range(0, num_samples, batch_size):
        take = min(batch_size, num_samples - start)
        key, pred_key, reconstruction_key = jax.random.split(key, 3)
        tokens, lp = predict(model.params, model.config, jax_batch, pred_key, n_atoms * 3, temperature)
        positions.append(np.asarray(model.decollate(tokens, lengths, key=reconstruction_key)[:take], dtype=np.float64))
        logq.append(np.asarray(lp[:take], dtype=np.float64))
    xyz = np.concatenate(positions)
    z = np.broadcast_to(atoms[None, :, None], (num_samples, n_atoms, 1))
    return SampleBatch(samples=np.concatenate((z, xyz), axis=-1), logq=np.concatenate(logq),
                       behavior_token_log_probs=None, behavior_token_ids=None)
