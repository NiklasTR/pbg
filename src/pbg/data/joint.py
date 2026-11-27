"""Turn a stored frame of the joint dataset into the frame that PBG-Joint trains on.

``pbg.models.pbg.loader.molecular_batches`` applies two steps to every frame:

1. ``select_representation``: a seeded choice of how the prompt writes the
   HELM. A cyclic peptide starts at a random residue, and 0 to 2 residues are
   written as inline SMILES tokens instead of library names. Stored cyclic
   prompts already write every residue inline.
2. ``fit_frame``: the most nearest waters whose tokens fit the budget. The
   conformer, the SMILES and the environment text are cut to match.

The stored atom order and coordinates are authoritative: step 1 changes only
how the prompt states them. The caches are bounded by the number of systems,
never by the number of frames.
"""

import hashlib
from dataclasses import replace
from functools import lru_cache

import numpy as np

from pbg.core.environment import environment_description
from pbg.core.helm import (
    join_peptide,
    load_helm_monomer_library,
    rewrite_helm_residues,
    rotate_cyclic_helm,
    split_peptide,
)


@lru_cache(maxsize=32768)
def helm_templates(helm: str):
    """Return ``(polymer, other HELM sections, named tokens, inline tokens)`` of a PEPTIDE HELM.

    Residue ``i`` can be written as its library name ``named[i]`` (for example
    ``A``) or as the inline SMILES token ``inline[i]``; both come from the HELM
    monomer library. None for a CHEM HELM.
    """
    if not helm.startswith('PEPTIDE'):
        return None
    library = load_helm_monomer_library()
    polymer, tokens, sections = split_peptide(helm)
    # Rewriting all sites resolves chemistry once. The selected output sites
    # later use string assembly, not repeated RDKit reconstruction.
    alternatives = split_peptide(rewrite_helm_residues(helm, list(range(1, len(tokens) + 1)), library))[1]
    if sections[0]:
        rotate_cyclic_helm(helm, 0, library)  # validate supported closure
    return polymer, tuple(sections), tuple(tokens), tuple(alternatives)


def select_representation(frame, seed: int):
    """Return the frame with the HELM its prompt shows, and the frame's seeded RNG.

    The RNG is seeded by ``seed`` and the frame key ``metadata['pilot_key']``. A
    cyclic peptide starts at a random residue. A random set of 0 to 2 residues
    is written inline. A CHEM HELM is not changed. The RNG is returned for the
    augmentation that follows.
    """
    identity = str(frame.metadata['pilot_key'])
    digest = hashlib.sha256(f'{seed}:{identity}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    templates = helm_templates(frame.helm)
    metadata = dict(frame.metadata)
    metadata.update(helm_rotation=0, helm_rewrite_selected=0, helm_rewrite_changed=0)
    if templates is None:
        return replace(frame, metadata=metadata), rng
    polymer, sections, named, inline = templates
    n = len(named)
    offset = int(rng.integers(n)) if sections[0] else 0
    count = int(rng.integers(min(2, n) + 1))
    selected = set(rng.choice(n, count, replace=False).tolist())
    tokens = [inline[i] if i in selected else named[i] for i in range(n)]
    changed = sum(named[i] != inline[i] for i in selected)
    tokens = tokens[offset:] + tokens[:offset]
    helm = join_peptide(polymer, tokens, sections)
    metadata.update(helm_rotation=offset, helm_rewrite_selected=count, helm_rewrite_changed=changed)
    return replace(frame, helm=helm, metadata=metadata), rng


@lru_cache(maxsize=4096)
def prefix_smiles(smiles: str, atoms: int) -> str:
    """Return the mapped SMILES of atoms ``1..atoms``: the solute and the waters kept by ``fit_frame``."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    selected = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() <= atoms]
    if len(selected) != atoms:
        raise ValueError('prefix graph does not match coordinate prefix')
    return Chem.MolFragmentToSmiles(mol, atomsToUse=selected, canonical=False)


def fit_frame(frame, config, tokeniser, budget: int):
    """Keep the solute and the most waters whose tokens fit in ``budget``, or return None.

    Stored waters are ordered by distance from the solute, so the first ``k``
    are the nearest. The conformer, SMILES and environment text (``"{k} water,
    ..."``) are cut to those ``k`` waters. A frame without waters is kept whole
    if it fits.
    """
    meta = frame.metadata
    solute = int(meta['n_solute_atoms'])
    waters = int(meta['n_waters'])
    fixed = 2
    helm_len = len(tokeniser.encode_helm(frame.helm))
    # A count's text length need not be monotonic for arbitrary BPE tokenizers.
    # Check feasible k descending; cache environment descriptions in tokenizer.
    upper = min(waters, max(0, (budget - fixed - helm_len - config.geom_width * solute) // (3 * config.geom_width)))
    candidates = range(upper, 0, -1) if waters else (0,)
    for k in candidates:
        environment = environment_description(meta['target_id'], k)
        prompt = tokeniser.encode_prompt(frame.helm, environment=environment, temperature_K=frame.temperature)
        atoms = solute + 3 * k
        cost = fixed + len(prompt) + config.geom_width * atoms
        if cost > budget:
            continue
        metadata = dict(meta, presented_waters=k, segment_length=cost)
        return replace(frame, environment=environment, conformer=frame.conformer[:atoms],
            smiles=frame.smiles if atoms == len(frame.conformer) else prefix_smiles(frame.smiles, atoms),
            energy=None if atoms != len(frame.conformer) else frame.energy, metadata=metadata)
    return None
