"""The molecular system a PBG samples: its prompt, its atoms and the checkpoint's rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pbg.checkpoint import load_pickle
from pbg.core.environment import PROTOCOLS, environment_description
from pbg.core.helm import mapped_smiles_to_chem_helm
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_helm, sequence_to_smiles
from pbg.core.tokeniser import VocabSchema, make_tokeniser

if TYPE_CHECKING:
    from pbg.models.pbg.model import PBG


@dataclass(frozen=True, slots=True)
class MolecularSystem:
    id: str
    mapped_smiles: str
    helm: str
    environment: str
    temperature_K: float
    sequence: str | None = None

    @classmethod
    def from_protocol(cls, *, id: str, mapped_smiles: str | None = None, helm: str,
                      target_id: str, temperature_K: float, n_waters: int = 0,
                      atom_order: str = "prompt_canonical_v1"):
        """Use training conditions; omitted SMILES compiles the checkpoint's traversal."""
        from rdkit import Chem

        environment = environment_description(target_id, n_waters)
        if mapped_smiles is None:
            from pbg.core.topology import PromptTopology
            mapped_smiles = PromptTopology.compile(helm, environment, atom_order).mapped_smiles
        mol = Chem.MolFromSmiles(mapped_smiles, sanitize=False)
        waters = 0
        for fragment in Chem.GetMolFrags(mol):
            atoms = [mol.GetAtomWithIdx(i) for i in fragment]
            if sorted(a.GetAtomicNum() for a in atoms) == [1, 1, 8] and all(a.GetFormalCharge() == 0 for a in atoms):
                waters += 1
        if PROTOCOLS[target_id].solvent == 'explicit_water' and waters != n_waters:
            raise ValueError('requested water count disagrees with mapped topology')
        return cls(id=id, mapped_smiles=mapped_smiles, helm=helm,
            environment=environment,
            temperature_K=float(temperature_K))

    @classmethod
    def from_peptide(
        cls,
        sequence: str,
        *,
        environment: str,
        temperature_K: float,
        id: str | None = None,
    ) -> "MolecularSystem":
        return cls(
            id=sequence if id is None else id,
            mapped_smiles=sequence_to_smiles(sequence),
            helm=sequence_to_helm(sequence),
            environment=environment,
            temperature_K=float(temperature_K),
            sequence=sequence,
        )

    @classmethod
    def from_smiles(
        cls,
        mapped_smiles: str,
        *,
        environment: str,
        temperature_K: float,
        id: str,
    ) -> "MolecularSystem":
        return cls(
            id=id,
            mapped_smiles=mapped_smiles,
            helm=mapped_smiles_to_chem_helm(mapped_smiles),
            environment=environment,
            temperature_K=float(temperature_K),
        )


@dataclass(frozen=True)
class SamplingPolicy:
    generation: str
    atom_order: str = 'prompt_canonical_v1'

    @classmethod
    def from_model(cls, model, requested: str = 'auto'):
        if requested not in {'auto', 'core', 'joint'}:
            raise ValueError(f'Unknown model generation: {requested}')
        name = model.config.vocab_name or ''
        # Core vocabularies (core_bin*) take a checkpoint-specific environment
        # prompt; the joint vocabulary takes the compact Kelvin prompt.
        generation = 'joint' if name.startswith('joint_') else 'core'
        if requested not in {'auto', generation}:
            raise ValueError(f'Requested {requested} generation but checkpoint uses {generation}')
        from pbg.core.topology import ATOM_ORDER_IMPLEMENTATIONS

        if generation == 'joint' and (
            'core_tokens' not in model.vocab
            or model.config.prompt_format != 'compact_kelvin_v1'
            or model.config.atom_order not in ATOM_ORDER_IMPLEMENTATIONS
        ):
            raise ValueError('Joint sampling requires its compact prompt and a prompt-derived traversal contract')
        return cls(generation, model.config.atom_order if generation == 'joint' else 'prompt_canonical_v1')

    def system(self, system, *, target_id: str | None = None, n_waters: int = 0):
        """Resolve physical conditions without replacing checkpoint vocabulary IDs."""
        if self.generation != 'joint':
            if target_id is not None:
                raise ValueError('Core checkpoints require their checkpoint-specific environment prompt')
            return system
        if target_id is not None:
            system = replace(system, environment=environment_description(target_id, n_waters))
        fields = [field.strip() for field in system.environment.split(',')]
        if len(fields) != 2 or fields[-1] not in PROTOCOLS:
            raise ValueError('Joint sampling requires a registered energy protocol, not a bare environment')
        from pbg.core.topology import PromptTopology

        topology = PromptTopology.compile(system.helm, system.environment, self.atom_order)
        expected = environment_description(fields[-1], topology.n_waters)
        if ''.join(expected.split()) != ''.join(system.environment.split()):
            raise ValueError('Joint environment and energy protocol disagree')
        if not topology.agrees_with(system.mapped_smiles):
            raise ValueError('Joint system differs from its prompt-derived atom traversal')
        return system


def checkpoint_environment_prompt(
    checkpoint: str | Path, *, label: str = "water"
) -> str:
    """Return the environment prompt text a checkpoint was trained with.

    The prompt string is embedded verbatim as BPE text by
    ``make_tokeniser(...).encode_prompt``, so sampling with a different wording
    than training silently produces a prompt-prefix mismatch. Training resolves
    the short stored label (for example ``"water"``) through the data source's
    ``environment_prompt`` map (see
    :func:`pbg.models.pbg.loader.iter_training_batches`); this
    function replays that lookup from the checkpoint metadata so evaluation
    scripts do not have to hard-code a wording that may not match.

    Raises ``KeyError`` when the checkpoint carries no usable metadata, so that
    a mis-prompted evaluation fails loudly rather than degrading silently.
    """
    path = Path(checkpoint)
    payload = load_pickle(path)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    explicit = metadata.get("environment_prompt") if isinstance(metadata, dict) else None
    if isinstance(explicit, dict) and label in explicit:
        return str(explicit[label])
    parameters = metadata.get("parameters") if isinstance(metadata, dict) else None
    data = parameters.get("data") if isinstance(parameters, dict) else None
    sources = data.get("sources") if isinstance(data, dict) else None
    if isinstance(sources, list) and any(source.get('molecular_single_pass') for source in sources):
        raise ValueError('Joint molecular training builds prompts from energy protocols; use SamplingPolicy.system')
    if not isinstance(sources, list) or not sources:
        raise KeyError(
            f"{path} has no data.sources metadata; pass the environment prompt "
            "explicitly after confirming the wording used during training"
        )
    prompts: dict[str, str] = {}
    for source in sources:
        mapping = source.get("environment_prompt") if isinstance(source, dict) else None
        if isinstance(mapping, dict):
            prompts.update({str(k): str(v) for k, v in mapping.items()})
    if not prompts:
        raise KeyError(
            f"{path} records no environment_prompt mapping; pass the environment "
            "prompt explicitly after confirming the wording used during training"
        )
    if label in prompts:
        return prompts[label]
    if len(prompts) == 1:
        return next(iter(prompts.values()))
    raise KeyError(
        f"{path} has no environment prompt for {label!r}; available: {sorted(prompts)}"
    )


def prompt_arrays(
    model: PBG,
    system: MolecularSystem,
    batch_size: int,
):
    """Prompt tokens and the atom and level index of each coordinate position.

    Returns ``(prompt, atom_index, level_index, prompt_len, coord_len, z)`` for
    ``batch_size`` copies of ``system``.
    """
    cfg = model.config
    if cfg.atom_order != "source_v1":
        from pbg.core.topology import PromptTopology
        topology = PromptTopology.compile(system.helm, system.environment, cfg.atom_order)
        if not topology.agrees_with(system.mapped_smiles):
            raise ValueError("Sampling topology does not match the checkpoint's prompt-derived atom traversal")
    schema = VocabSchema.of(model.vocab)
    if cfg.conditioning == "atom":
        # Atom conditioning has no text prompt: the segment is [BOS | coords].
        prompt_ids = np.zeros((0,), dtype=np.int32)
    else:
        prompt_ids = make_tokeniser(model.vocab, prompt_format=cfg.prompt_format).encode_prompt(
            system.helm,
            environment=system.environment,
            temperature_K=system.temperature_K,
        )
    z = atom_numbers_from_smiles(system.mapped_smiles).astype(np.float32)
    n_atoms = int(z.size)
    prompt_len = 1 + int(prompt_ids.size)
    depth = cfg.geom_width
    coord_len = n_atoms * depth
    if prompt_len + coord_len >= cfg.max_seq_len:
        raise ValueError("sample prompt exceeds config.max_seq_len")
    prompt = np.full((batch_size, prompt_len), schema.bos_id, dtype=np.int32)
    atom_index = np.zeros((batch_size, cfg.max_seq_len), dtype=np.int32)
    level_index = np.zeros_like(atom_index)
    prompt[:, 1:] = prompt_ids[None, :]
    atoms = np.repeat(np.arange(1, n_atoms + 1, dtype=np.int32), depth)
    levels = np.tile(np.arange(1, depth + 1, dtype=np.int32), n_atoms)
    atom_index[:, prompt_len : prompt_len + coord_len] = atoms[None, :]
    level_index[:, prompt_len : prompt_len + coord_len] = levels[None, :]
    return prompt, atom_index, level_index, prompt_len, coord_len, z
