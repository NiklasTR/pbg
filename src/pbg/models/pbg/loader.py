"""Training data for PBG: frame preparation, collation, the shard cursor and loader workers.

This module imports no JAX. Spawned data-loader processes start in
:func:`pbg_batch_worker`, which hides the accelerators before any batch work:
a child that initialises a GPU beside the trainer can stall or abort ranks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pbg.checkpoint import retire
from pbg.core.augmenter import (
    AugmentationBoundsError,
    Augmenter,
    apply_augmenter,
    init_augmenter,
)
from pbg.core.frame import AtomConditionedFrame, DataFrame, TokenisedFrame
from pbg.core.helm import mapped_smiles_to_chem_helm
from pbg.core.peptide import resolve_linear_peptide_helm
from pbg.core.tokeniser import (
    QwenJointTokeniser,
    VocabSchema,
    apply_tokeniser_conformer,
    make_tokeniser,
)
from pbg.data.joint import fit_frame, select_representation
from pbg.data.webdataset import iter_frames_from_tar
from pbg.models.pbg.config import PBGConfig

if TYPE_CHECKING:
    from pbg.models.pbg.model import PBG

log = logging.getLogger(__name__)

_FIRST_BATCH_DEADLINE_S = float(os.environ.get("PBG_FIRST_BATCH_DEADLINE_S", "1200"))
_MP_CTX = mp.get_context("spawn")
_PREPARE_RNG_STRIDE = 10007


# ── Shard cursor ───────────────────────────────────────────────────


CURSOR_VERSION = 1
BATCH_DONE_PASS_KEY = "_cursor_done_pass"
BATCH_DONE_SHARD_KEY = "_cursor_done_shard"
BATCH_MOLECULAR_KEY = "_cursor_molecular"
# Batches written by the multi-source loader also carry a source index.
CURSOR_BATCH_KEYS = ("_cursor_source", BATCH_DONE_PASS_KEY, BATCH_DONE_SHARD_KEY, BATCH_MOLECULAR_KEY)


def list_source_tar_paths(source: dict[str, Any]) -> tuple[Path, ...]:
    if source.get('molecular_single_pass'):
        root = Path(source['dir'])
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        return tuple(root / Path(s['path']).name for s in manifest['shards'])
    data_dir = Path(str(source["dir"]))
    return tuple(sorted(data_dir.rglob("*.tar")))


def tar_fingerprint(data_dir: Path, tar_paths: tuple[Path, ...]) -> str:
    resolved_root = data_dir.resolve()
    rels: list[str] = []
    for path in tar_paths:
        try:
            rels.append(path.resolve().relative_to(resolved_root).as_posix())
        except ValueError:
            rels.append(path.name)
    return hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()


def catch_up_shard_work(
    shard_pass: tuple[int, ...] | list[int] | np.ndarray,
) -> list[tuple[int, int]]:
    passes = tuple(int(value) for value in shard_pass)
    n_shards = len(passes)
    if n_shards == 0:
        return []
    min_pass = min(passes)
    max_pass = max(passes)
    due: list[tuple[int, int]] = []
    for pass_id in range(min_pass, max_pass + 1):
        for shard_idx in range(n_shards):
            if passes[shard_idx] == pass_id:
                due.append((pass_id, shard_idx))
    return due


def iter_assigned_tar_shards(
    tar_paths: tuple[Path, ...],
    *,
    worker_id: int,
    num_workers: int,
    shard_pass: tuple[int, ...] | list[int] | np.ndarray | None = None,
) -> Iterator[tuple[int, int, Path]]:
    n_shards = len(tar_paths)
    if n_shards == 0:
        return
    if worker_id < 0 or num_workers <= 0 or worker_id >= num_workers:
        raise ValueError("worker id must satisfy 0 <= worker_id < num_workers")
    passes = (
        tuple(int(value) for value in shard_pass)
        if shard_pass is not None
        else (0,) * n_shards
    )
    if len(passes) != n_shards:
        raise ValueError(
            f"shard_pass length {len(passes)} does not match tar count {n_shards}"
        )
    due = catch_up_shard_work(passes)
    tail = list(range(n_shards))[worker_id::num_workers]
    if not due[worker_id::num_workers] and not tail:
        return
    for pass_id, shard_idx in due[worker_id::num_workers]:
        yield pass_id, shard_idx, tar_paths[shard_idx]
    pass_id = max(passes) + 1
    while True:
        if not tail:
            return
        for shard_idx in tail:
            yield pass_id, shard_idx, tar_paths[shard_idx]
        pass_id += 1


def split_cursor_batch(
    batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = {key: batch[key] for key in CURSOR_BATCH_KEYS if key in batch}
    clean = {key: value for key, value in batch.items() if key not in CURSOR_BATCH_KEYS}
    return clean, meta


def parse_saved_cursor(raw: Any) -> dict[str, Any] | None:
    """Return the per-source shard passes and molecular cursor of a saved cursor.

    Cursors saved by the multi-source loader also hold a mixing RNG state per
    process; one source never draws from it, so it is ignored.
    """
    if not isinstance(raw, dict) or raw.get("version") != CURSOR_VERSION:
        return None
    sources = raw.get("sources")
    if not isinstance(sources, dict):
        return None
    parsed: dict[str, Any] = {}
    for key, spec in sources.items():
        shard_pass = spec.get("shard_pass") if isinstance(spec, dict) else None
        if not isinstance(shard_pass, list) or not all(isinstance(item, (int, np.integer)) for item in shard_pass):
            return None
        parsed[str(key)] = {
            "fingerprint": str(spec.get("fingerprint", "")),
            "n_shards": int(spec.get("n_shards", len(shard_pass))),
            "shard_pass": [int(item) for item in shard_pass],
        }
    return {"sources": parsed, "molecular": raw.get("molecular")}


@dataclass
class DataCursorTracker:
    """Pass count of every tar shard of the training source, for resume."""

    key: str
    fingerprint: str
    shard_pass: np.ndarray
    tar_paths: tuple[Path, ...]
    molecular: dict[str, Any] | None = None

    def note_trained_batch(self, meta: dict[str, Any]) -> None:
        if BATCH_MOLECULAR_KEY in meta:
            self.molecular = dict(meta[BATCH_MOLECULAR_KEY])
        passes = np.asarray(meta.get(BATCH_DONE_PASS_KEY, []), dtype=np.int32).reshape(-1)
        shards = np.asarray(meta.get(BATCH_DONE_SHARD_KEY, []), dtype=np.int32).reshape(-1)
        if passes.size != shards.size:
            return
        for pass_id, shard_idx in zip(passes.tolist(), shards.tolist(), strict=True):
            if 0 <= int(shard_idx) < self.shard_pass.size:
                current = int(self.shard_pass[int(shard_idx)])
                self.shard_pass[int(shard_idx)] = max(current, int(pass_id) + 1)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "version": CURSOR_VERSION,
            "sources": {
                self.key: {
                    "fingerprint": self.fingerprint,
                    "n_shards": int(self.shard_pass.size),
                    "shard_pass": self.shard_pass.astype(np.int64).tolist(),
                }
            },
            "molecular": self.molecular,
        }


def make_data_cursor_tracker(data: dict[str, Any], *, saved: Any = None) -> DataCursorTracker:
    """Start the cursor of ``data``'s single source, restored from ``saved`` if it matches."""
    (source,) = data["sources"]
    parsed = parse_saved_cursor(saved)
    if saved is not None and parsed is None:
        log.warning("checkpoint data_cursor is missing or unreadable; data streams restart at shard 0")
    key = str(source.get("name") or source["dir"])
    tar_paths = list_source_tar_paths(source)
    fingerprint = tar_fingerprint(Path(str(source["dir"])), tar_paths) if tar_paths else ""
    restored = parsed["sources"].get(key) if parsed is not None else None
    if (
        restored is not None
        and restored["fingerprint"] == fingerprint
        and int(restored["n_shards"]) == len(tar_paths)
        and len(restored["shard_pass"]) == len(tar_paths)
    ):
        shard_pass = np.asarray(restored["shard_pass"], dtype=np.int64)
    else:
        if restored is not None and not source.get("molecular_append_only"):
            log.warning(
                "data_cursor source %s does not match the tar tree; restarting that source at shard 0",
                key,
            )
        shard_pass = np.zeros(len(tar_paths), dtype=np.int64)
    return DataCursorTracker(
        key=key,
        fingerprint=fingerprint,
        shard_pass=shard_pass,
        tar_paths=tar_paths,
        molecular=parsed["molecular"] if parsed is not None else None,
    )


# ── Frame preparation ──────────────────────────────────────────────


def clip_solvent_to_atom_budget(
    frame: DataFrame,
    max_atoms: int,
) -> DataFrame | None:
    """Retain the solute and as many complete water molecules as fit."""
    if frame.conformer is None:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(frame.smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {frame.smiles}")
    fragments = Chem.GetMolFrags(mol, sanitizeFrags=False)
    fragment_sizes = np.fromiter(
        (len(fragment) for fragment in fragments),
        dtype=np.int32,
        count=len(fragments),
    )
    atom_numbers = np.fromiter(
        (atom.GetAtomicNum() for atom in mol.GetAtoms()),
        dtype=np.int16,
        count=mol.GetNumAtoms(),
    )
    formal_charges = np.fromiter(
        (atom.GetFormalCharge() for atom in mol.GetAtoms()),
        dtype=np.int16,
        count=mol.GetNumAtoms(),
    )
    fragment_atoms = np.concatenate(fragments)
    fragment_ids = np.empty(mol.GetNumAtoms(), dtype=np.int32)
    fragment_ids[fragment_atoms] = np.repeat(
        np.arange(len(fragments), dtype=np.int32),
        fragment_sizes,
    )
    oxygen_counts = np.bincount(
        fragment_ids,
        weights=atom_numbers == 8,
        minlength=len(fragments),
    )
    hydrogen_counts = np.bincount(
        fragment_ids,
        weights=atom_numbers == 1,
        minlength=len(fragments),
    )
    charge_magnitudes = np.bincount(
        fragment_ids,
        weights=np.abs(formal_charges),
        minlength=len(fragments),
    )
    is_water = (
        (fragment_sizes == 3)
        & (oxygen_counts == 1)
        & (hydrogen_counts == 2)
        & (charge_magnitudes == 0)
    )
    if bool(np.all(is_water)):
        is_water[0] = False

    mandatory_atoms = int(fragment_sizes[~is_water].sum())
    if mandatory_atoms > max_atoms:
        return None
    water_indices = np.flatnonzero(is_water)
    water_sizes = fragment_sizes[water_indices]
    num_water = int(
        np.searchsorted(
            np.cumsum(water_sizes),
            max_atoms - mandatory_atoms,
            side="right",
        )
    )
    selected_fragments = ~is_water
    selected_fragments[water_indices[:num_water]] = True
    selected_atoms = fragment_atoms[selected_fragments[fragment_ids[fragment_atoms]]]

    atom_maps = np.fromiter(
        (atom.GetAtomMapNum() for atom in mol.GetAtoms()),
        dtype=np.int32,
        count=mol.GetNumAtoms(),
    )
    selected_maps = atom_maps[selected_atoms]
    if (
        np.any(selected_maps <= 0)
        or np.unique(selected_maps).size != selected_maps.size
        or int(selected_maps.max()) > frame.conformer.shape[0]
    ):
        raise ValueError("solvent clipping requires unique 1-based atom maps")
    conformer_indices = np.sort(selected_maps - 1)
    clipped_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=selected_atoms.tolist(),
        canonical=False,
    )
    return replace(
        frame,
        conformer=frame.conformer[conformer_indices],
        smiles=clipped_smiles,
    )


def prepare_frame(
    frame: DataFrame,
    config: PBGConfig,
    vocab: dict[str, object],
    augmenter: Augmenter,
    *,
    rng: np.random.Generator,
    derive_chem_helm_from_smiles: bool = False,
    prune_water_from_chem_helm: bool = False,
    clip_solvent_to_fit: bool = False,
    prompt_notation: str | None = None,
    tokeniser: QwenJointTokeniser | None = None,
) -> TokenisedFrame | None:
    """Augment, prompt and tokenise one frame for training; ``None`` drops it."""
    if frame.conformer is None:
        return None
    if config.conditioning == "atom":
        # ROBIN's features; imported here because the module brings in JAX.
        from pbg.models.pbg.robin import atom_features as _atom_features
    helm: str | None = None
    helm_ids: np.ndarray | None = None
    helm_smiles: str | None = None

    if prompt_notation not in {None, "PEPTIDE", "CHEM"}:
        raise ValueError(
            f"prompt_notation must be 'PEPTIDE' or 'CHEM', got {prompt_notation!r}"
        )
    use_chem_helm = (
        derive_chem_helm_from_smiles
        if prompt_notation is None
        else prompt_notation == "CHEM"
    )
    # Atom conditioning replaces the text prompt with ROBIN's per-atom
    # features, so the segment is [BOS | coords | EOS].
    atom_conditioned = config.conditioning == "atom"
    if atom_conditioned and (use_chem_helm or clip_solvent_to_fit):
        raise ValueError("atom conditioning supports only linear peptides without solvent")

    def derive_helm(current: DataFrame) -> str:
        if use_chem_helm:
            return mapped_smiles_to_chem_helm(
                current.smiles,
                prune_water=prune_water_from_chem_helm,
            )
        return resolve_linear_peptide_helm(
            helm=current.helm,
            sequence=current.sequence,
            smiles=current.smiles,
        )

    def tokenise_helm(current: DataFrame, helm: str) -> np.ndarray:
        return (tokeniser if tokeniser is not None else make_tokeniser(vocab, prompt_format=config.prompt_format)).encode_prompt(
            helm,
            environment=current.environment,
            temperature_K=current.temperature,
        )

    if clip_solvent_to_fit:
        preliminary_helm = derive_helm(frame)
        preliminary_ids = tokenise_helm(frame, preliminary_helm)
        helm = preliminary_helm
        helm_ids = preliminary_ids
        helm_smiles = frame.smiles
        max_atoms = (
            config.max_seq_len
            - 2
            - int(preliminary_ids.shape[0])
        ) // config.geom_width
        if frame.conformer.shape[0] > max_atoms:
            frame = clip_solvent_to_atom_budget(frame, max_atoms)
            if frame is None:
                return None
            if use_chem_helm and not prune_water_from_chem_helm:
                helm = None
                helm_ids = None
                helm_smiles = None
    try:
        frame = apply_augmenter(augmenter, frame, rng=rng)
    except AugmentationBoundsError:
        log.warning(
            "skipping out-of-box frame id=%s frame_id=%s "
            "reason=rotation_attempts_exhausted",
            frame.id,
            frame.frame_id,
        )
        return None
    if frame.conformer is None:
        return None

    while True:
        if helm is None or helm_ids is None or helm_smiles != frame.smiles:
            helm = None if atom_conditioned else derive_helm(frame)
            helm_ids = (
                np.zeros((0,), dtype=np.int32)
                if atom_conditioned
                else tokenise_helm(frame, helm)
            )
            helm_smiles = frame.smiles
        if not clip_solvent_to_fit:
            break
        max_atoms = (
            config.max_seq_len
            - 2
            - int(helm_ids.shape[0])
        ) // config.geom_width
        if frame.conformer.shape[0] <= max_atoms:
            break
        previous_atoms = int(frame.conformer.shape[0])
        frame = clip_solvent_to_atom_budget(frame, max_atoms)
        if frame is None:
            return None
        if frame.conformer.shape[0] >= previous_atoms:
            return None
    if config.atom_order != "source_v1":
        from pbg.core.topology import PromptTopology

        frame = PromptTopology.compile(helm, frame.environment or "vacuum", config.atom_order).align_frame(frame)
    dec_ids, dec_atom, dec_slot = apply_tokeniser_conformer(
        frame,
        method="bin",
        num_bins=config.bin_tokens,
        min_val=config.min_val,
        max_val=config.max_val,
    )
    n_atoms = int(frame.conformer.shape[0])
    helm_len = int(helm_ids.shape[0])
    seg_len = (
        2
        + helm_len
        + n_atoms * config.geom_width
    )
    frame_type, extra = (
        (AtomConditionedFrame,
         {"atom_features": _atom_features(frame.sequence, frame.conformer[:, 0])})
        if atom_conditioned
        else (TokenisedFrame, {})
    )
    return frame_type(
        **extra,
        seg_len=seg_len,
        helm_ids=np.asarray(helm_ids, dtype=np.int32),
        coord_ids=np.asarray(dec_ids, dtype=np.int32),
        coord_atom=np.asarray(dec_atom, dtype=np.int32),
        coord_slot=np.asarray(dec_slot, dtype=np.int32),
        environment=None if frame.environment is None else str(frame.environment),
        temperature=None if frame.temperature is None else float(frame.temperature),
        energy=None if frame.energy is None else float(frame.energy),
        logq=None if frame.logq is None else float(frame.logq),
        id=frame.id,
        frame_id=frame.frame_id,
        helm=helm,
        smiles="" if frame.smiles is None else str(frame.smiles),
        sequence=frame.sequence,
        self_distillation_weight=frame.self_distillation_weight,
        behavior_token_log_probs=frame.behavior_token_log_probs,
        behavior_token_ids=frame.behavior_token_ids,
        frame_valid=frame.frame_valid,
        source=frame.source,
        generator_checkpoint=frame.generator_checkpoint,
        alpha=frame.alpha,
        epsilon=frame.epsilon,
        time=frame.time,
        metadata=frame.metadata,
    )


def augmenter_from_config(config: PBGConfig) -> Augmenter:
    translate = config.translate if config.translate is not None else config.augment
    return init_augmenter(
        center="mass" if config.center and config.center_of_mass else config.center,
        rotate=config.augment,
        translate=translate,
        min_val=config.min_val,
        max_val=config.max_val,
    )


# ── Collation ──────────────────────────────────────────────────────


def _pad_sample_vectors(
    cfg: PBGConfig,
    batch_size: int,
    values: list[float],
    *,
    pad_value: float = np.nan,
) -> np.ndarray:
    capacity = cfg.max_samples_per_batch(batch_size)
    n = len(values)
    if n > capacity:
        raise ValueError(
            f"collated batch has {n} samples but max_samples_per_batch is {capacity} "
            f"(max_samples_factor={cfg.max_samples_factor}, batch_size={batch_size}, "
            f"max_seq_len={cfg.max_seq_len})"
        )
    out = np.full((capacity,), pad_value, dtype=np.float32)
    if n:
        out[:n] = np.asarray(values, dtype=np.float32)
    return out


def collate_tokenised_frames(
    cfg: PBGConfig,
    vocab: dict[str, object],
    frames: list[list[TokenisedFrame]],
) -> dict[str, np.ndarray]:
    schema = VocabSchema.of(vocab)
    bos_id = schema.bos_id
    eos_id = schema.terminator_id
    pad_id = schema.pad_id
    coord_base = schema.coord_base_id
    max_len = cfg.max_seq_len
    tokens = np.full((len(frames), max_len), pad_id, dtype=np.int32)
    atom_index = np.zeros_like(tokens)
    level_index = np.zeros_like(tokens)
    segment_id = np.full_like(tokens, -1)
    sample_id = np.full_like(tokens, -1)
    sample_energy: list[float] = []
    sample_temperature: list[float] = []
    sample_logq: list[float] = []
    sample_alpha: list[float] = []
    sample_self_distillation_weight: list[float] = []
    sample_frame_valid: list[float] = []
    # Atom conditioning: a per-row table of atom features, the sample that owns
    # each atom slot, and for each position the slot of the atom it predicts.
    atom_conditioned = cfg.conditioning == "atom"
    n_features = 0
    if atom_conditioned:
        from pbg.models.pbg.robin import FEATURES

        n_features = len(FEATURES)
    atom_features = np.zeros(
        (len(frames), cfg.max_conditioned_atoms, n_features), dtype=np.int32
    )
    atom_sample = np.full(atom_features.shape[:2], -1, dtype=np.int32)
    atom_slot = np.full_like(tokens, -1)

    for row_idx, group in enumerate(frames):
        offset = 0
        row_atoms = 0
        for seg_idx, frame in enumerate(group):
            current_sample_id = len(sample_energy)
            sample_energy.append(
                float(frame.energy) if frame.energy is not None else np.nan
            )
            sample_temperature.append(
                float(frame.temperature) if frame.temperature is not None else np.nan
            )
            sample_logq.append(float(frame.logq) if frame.logq is not None else np.nan)
            sample_alpha.append(float(frame.alpha) if frame.alpha is not None else np.nan)
            sample_self_distillation_weight.append(
                float(frame.self_distillation_weight)
            )
            sample_frame_valid.append(float(frame.frame_valid))
            helm_len = int(frame.helm_ids.shape[0])
            if offset + frame.seg_len > max_len:
                raise ValueError("packed frame group exceeds config.max_seq_len")

            row_t = tokens[row_idx]
            row_a = atom_index[row_idx]
            row_l = level_index[row_idx]
            row_s = segment_id[row_idx]
            row_sample = sample_id[row_idx]
            row_t[offset] = bos_id
            row_s[offset] = seg_idx
            row_sample[offset] = current_sample_id
            prompt_start = offset + 1
            helm_stop = prompt_start + helm_len
            row_t[prompt_start:helm_stop] = frame.helm_ids
            row_s[prompt_start:helm_stop] = seg_idx
            row_sample[prompt_start:helm_stop] = current_sample_id
            c0 = helm_stop
            coord_len = int(frame.coord_ids.shape[0])
            row_t[c0 : c0 + coord_len] = frame.coord_ids + coord_base
            row_a[c0 : c0 + coord_len] = frame.coord_atom + 1
            row_l[c0 : c0 + coord_len] = frame.coord_slot + 1
            row_s[c0 : c0 + coord_len] = seg_idx
            row_sample[c0 : c0 + coord_len] = current_sample_id
            row_t[c0 + coord_len] = eos_id
            row_s[c0 + coord_len] = seg_idx
            row_sample[c0 + coord_len] = current_sample_id
            offset = c0 + coord_len + 1
            if atom_conditioned:
                if not isinstance(frame, AtomConditionedFrame):
                    raise TypeError("atom conditioning requires AtomConditionedFrame inputs")
                n_atoms = int(frame.atom_features.shape[0])
                if row_atoms + n_atoms > cfg.max_conditioned_atoms:
                    raise ValueError("packed atoms exceed config.max_conditioned_atoms")
                atoms = slice(row_atoms, row_atoms + n_atoms)
                atom_features[row_idx, atoms] = frame.atom_features
                atom_sample[row_idx, atoms] = current_sample_id
                # The position before each coordinate token predicts it.
                atom_slot[row_idx, c0 - 1 : c0 - 1 + coord_len] = (
                    row_atoms + frame.coord_atom
                )
                row_atoms += n_atoms

    targets = np.full_like(tokens, pad_id)
    targets[:, :-1] = tokens[:, 1:]
    same_segment = np.zeros(tokens.shape, dtype=bool)
    same_segment[:, :-1] = (segment_id[:, :-1] >= 0) & (
        segment_id[:, :-1] == segment_id[:, 1:]
    )
    loss_mask = ((targets != pad_id) & same_segment).astype(np.float32)
    # Train only on coordinate tokens plus (unless ``loss_terminator`` is off)
    # the coordinate-block terminator; every prompt token (legacy structural
    # ids or joint Qwen BPE text) is masked out.
    terminator = (schema.terminator_id,) if cfg.loss_terminator else ()
    trainable = np.asarray((*schema.coord_ids, *terminator), dtype=np.int32)
    loss_mask *= np.isin(targets, trainable).astype(np.float32)
    post_coord_mask = loss_mask * np.isin(
        targets, np.asarray(schema.coord_ids, dtype=np.int32)
    ).astype(np.float32)
    batch_size = int(tokens.shape[0])
    energy = _pad_sample_vectors(cfg, batch_size, sample_energy)
    temperature = _pad_sample_vectors(cfg, batch_size, sample_temperature)
    logq = _pad_sample_vectors(cfg, batch_size, sample_logq)
    sample_valid = np.zeros((cfg.max_samples_per_batch(batch_size),), dtype=np.float32)
    sample_valid[: len(sample_energy)] = 1.0
    out = {
        "tokens": tokens,
        "targets": targets,
        "loss_mask": loss_mask,
        "post_coord_mask": post_coord_mask,
        "atom_index": atom_index,
        "level_index": level_index,
        "segment_id": segment_id,
        "sample_id": sample_id,
        "sample_energy": energy,
        "sample_temperature": temperature,
        "sample_logq": logq,
        "sample_alpha": _pad_sample_vectors(cfg, batch_size, sample_alpha),
        "sample_valid": sample_valid,
    }
    explicit_weight = _pad_sample_vectors(
        cfg, batch_size, sample_self_distillation_weight
    )
    explicit_valid = _pad_sample_vectors(
        cfg, batch_size, sample_frame_valid, pad_value=0.0
    )
    active = (sample_valid > 0) & (explicit_valid > 0)
    weights = np.where(active, explicit_weight, 0.0).astype(np.float32)
    sample_weight = np.zeros(tokens.shape, dtype=np.float32)
    # Only real token positions need metadata weights; padding is already zero.
    for sample_idx in range(len(sample_energy)):
        sample_weight[sample_id == sample_idx] = float(weights[sample_idx])
    out["sample_weight"] = sample_weight
    out["sample_self_distillation_weight"] = weights
    out["frame_valid"] = explicit_valid
    if atom_conditioned:
        out.update(
            atom_features=atom_features, atom_sample=atom_sample, atom_slot=atom_slot
        )
    return out


# ── Streaming loader ───────────────────────────────────────────────


def _loader_heartbeat(message: str, *args: object) -> None:
    """Emit loader progress even when the application never configures logging.

    Training entry points often leave the root logger at WARNING, which would
    silence the data-loader info channel entirely. Route through the logger
    and fall back to a timestamped stderr line so startup stalls stay visible.
    """
    log.info(message, *args)
    if not logging.getLogger().handlers and log.getEffectiveLevel() > logging.INFO:
        print(f"[dataloader {time.strftime('%H:%M:%S')}] " + message % args,
              flush=True, file=sys.stderr)


def pack_rows(
    model: PBG | PBGConfig,
    frames: Iterator[TokenisedFrame],
    source: dict[str, Any],
) -> Iterator[list[TokenisedFrame]]:
    config = getattr(model, "config", model)
    max_tokens = config.max_seq_len
    packing = str(source.get("packing", "greedy"))
    if packing not in {"greedy", "single"}:
        raise ValueError(f"unknown packing mode: {packing!r}")

    pending: TokenisedFrame | None = None
    frame_iter = iter(frames)
    while True:
        if pending is None:
            try:
                frame = next(frame_iter)
            except StopIteration:
                return
        else:
            frame = pending
            pending = None
        cost = frame.seg_len
        if cost > max_tokens:
            continue
        if packing == "single":
            yield [frame]
            continue
        group = [frame]
        group_tokens = cost
        while True:
            try:
                frame = next(frame_iter)
            except StopIteration:
                break
            cost = frame.seg_len
            if cost > max_tokens:
                continue
            if group_tokens + cost > max_tokens:
                pending = frame
                break
            group.append(frame)
            group_tokens += cost
        yield group


def _iter_prepared_frames_for_tar(
    source: dict[str, Any],
    path: Path,
    config: PBGConfig,
    *,
    seed: int,
    shard_idx: int,
    vocab: dict[str, object],
    augmenter: Augmenter,
    tokeniser: QwenJointTokeniser | None = None,
) -> Iterator[TokenisedFrame]:
    """Prepared frames of one tar shard, with the source's environment prompt."""
    prompts = source["environment_prompt"]
    temperature = source.get("temperature", source.get("temperature_K"))
    frame_idx = 0
    if tokeniser is None:
        tokeniser = make_tokeniser(vocab, prompt_format=config.prompt_format)
    for frame in iter_frames_from_tar(
        path,
        adapter=str(source.get("adapter", "many_peptides")),
        on_decode_error=str(source.get("on_decode_error", "raise")),
        temperature=None if temperature is None else float(temperature),
        sample_layout=source.get("sample_layout"),
        frame_id_below=source.get("frame_id_below"),
        sequence_allowlist=source.get("sequence_allowlist"),
    ):
        label = None if frame.environment is None else str(frame.environment)
        if label is None and len(prompts) == 1:
            prompt = next(iter(prompts.values()))
        elif label in prompts:
            prompt = prompts[label]
        else:
            raise KeyError(f"unknown stored environment {label!r}; available prompts: {sorted(prompts)}")
        frame = replace(frame, environment=str(prompt))
        # Two increments per frame: the trained checkpoints used these
        # augmentation seeds, so the stride is kept for reproducibility.
        frame_idx += 1
        frame_idx += 1
        prepared = prepare_frame(
            frame,
            config,
            vocab,
            augmenter,
            rng=np.random.default_rng(seed + _PREPARE_RNG_STRIDE * (shard_idx + 1) + frame_idx),
            derive_chem_helm_from_smiles=bool(
                source.get("derive_chem_helm_from_smiles", False)
            ),
            prune_water_from_chem_helm=bool(
                source.get("prune_water_from_chem_helm", False)
            ),
            clip_solvent_to_fit=bool(
                source.get(
                    "clip_solvent_to_fit",
                    source.get("derive_chem_helm_from_smiles", False),
                )
            ),
            prompt_notation=source.get("prompt_notation"),
            tokeniser=tokeniser,
        )
        if prepared is not None:
            yield prepared


def _iter_tagged_collated_batches(
    source: dict[str, Any],
    *,
    worker_id: int,
    num_workers: int,
    global_worker_offset: int | None,
    global_num_workers: int | None,
    seed: int,
    batch_size: int,
    config: PBGConfig,
    vocab: dict[str, object],
    augmenter: Augmenter,
    shard_pass: tuple[int, ...] | None,
) -> Iterator[dict[str, np.ndarray]]:
    """Collated batches of one loader worker's tar shards, tagged with the shards they finish."""
    tar_paths = list_source_tar_paths(source)
    if not tar_paths:
        raise FileNotFoundError(f"no .tar files found in {source.get('dir')}")
    effective_worker_id = worker_id if global_worker_offset is None else int(global_worker_offset) + worker_id
    effective_num_workers = num_workers if global_num_workers is None else int(global_num_workers)
    if not 0 <= effective_worker_id < effective_num_workers:
        raise ValueError(
            f"worker id {effective_worker_id} must lie in [0, {effective_num_workers})"
        )
    passes = shard_pass if shard_pass is not None else (0,) * len(tar_paths)
    due = catch_up_shard_work(passes)
    tail = list(range(len(tar_paths)))[effective_worker_id::effective_num_workers]
    if not due[effective_worker_id::effective_num_workers] and not tail:
        raise FileNotFoundError(
            f"decode worker {effective_worker_id + 1}/{effective_num_workers} has no tar files in {source.get('dir')}"
        )
    log.info(
        "decode worker %d/%d (local %d/%d): %d/%d tar files in %s",
        effective_worker_id + 1,
        effective_num_workers,
        worker_id + 1,
        num_workers,
        len(tail),
        len(tar_paths),
        source.get("dir"),
    )
    # Hold at most one batch plus a lookahead row. The old implementation
    # materialized every prepared row in a shard before yielding any batch.
    pending_rows: list[list[TokenisedFrame]] = []
    pending_done: list[tuple[int, int]] = []
    tokeniser = make_tokeniser(vocab, prompt_format=config.prompt_format)
    for pass_id, shard_idx, path in iter_assigned_tar_shards(
        tar_paths,
        worker_id=effective_worker_id,
        num_workers=effective_num_workers,
        shard_pass=passes,
    ):
        rows = iter(pack_rows(
            config,
            _iter_prepared_frames_for_tar(
                source,
                path,
                config,
                seed=seed,
                shard_idx=shard_idx,
                vocab=vocab,
                augmenter=augmenter,
                tokeniser=tokeniser,
            ),
            source,
        ))
        while True:
            row = next(rows, None)
            if row is None:
                # EOF is known only after lookahead. A queued batch must not
                # claim this shard until it includes the shard's last row.
                pending_done.append((pass_id, shard_idx))
            if len(pending_rows) == batch_size:
                batch = collate_tokenised_frames(config, vocab, pending_rows)
                done = list(dict.fromkeys(pending_done))
                batch[BATCH_DONE_PASS_KEY] = np.asarray([item[0] for item in done], dtype=np.int32).reshape(-1)
                batch[BATCH_DONE_SHARD_KEY] = np.asarray([item[1] for item in done], dtype=np.int32).reshape(-1)
                pending_rows.clear()
                pending_done.clear()
                yield batch
            if row is None:
                break
            pending_rows.append(row)


def iter_training_batches(
    model: PBG,
    data: dict[str, Any],
    *,
    seed: int,
    start_step: int = 0,
    process_index: int = 0,
    process_count: int = 1,
    data_cursor: DataCursorTracker | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield collated training batches of ``data``'s single source, endlessly."""
    (source,) = data["sources"]
    if source.get("molecular_single_pass"):
        if process_count != 1:
            raise ValueError("deterministic molecular pilot supports one host only")
        yield from molecular_batches(model, source, batch_size=int(data["batch_size"]),
            seed=seed, start_step=start_step, cursor_tracker=data_cursor)
        return
    batch_size = int(data["batch_size"])
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive")
    if process_index < 0 or process_count <= 0 or process_index >= process_count:
        raise ValueError("process sharding requires 0 <= process_index < process_count")
    data_loader_workers = int(data.get("data_loader_workers", 1))
    prefetch = int(data.get("prefetch", 16))
    if data_loader_workers <= 0 or prefetch <= 0:
        raise ValueError("data_loader_workers and prefetch must be positive")
    tracker = data_cursor if data_cursor is not None else make_data_cursor_tracker(data)
    log.info("data loader workers=%d prefetch=%d source=%s", data_loader_workers, prefetch, tracker.key)
    shard_pass = tuple(int(value) for value in tracker.shard_pass.tolist())
    # The paper runs drew this per-source seed; keep it for their data order.
    seed += 1009
    augmenter = augmenter_from_config(model.config)
    vocab = model.vocab
    config = model.config
    global_worker_offset = int(process_index) * data_loader_workers
    global_num_workers = int(process_count) * data_loader_workers
    if data_loader_workers == 1:
        yield from _iter_tagged_collated_batches(
            source,
            worker_id=0,
            num_workers=1,
            global_worker_offset=global_worker_offset,
            global_num_workers=global_num_workers,
            seed=seed,
            batch_size=batch_size,
            config=config,
            vocab=vocab,
            augmenter=augmenter,
            shard_pass=shard_pass,
        )
        return
    batch_queue: mp.Queue[Any] = _MP_CTX.Queue(maxsize=prefetch)
    procs: list[mp.Process] = []
    t_spawn = time.monotonic()
    log.info(
        "spawning %d data-loader processes via %s",
        data_loader_workers,
        pbg_batch_worker.__module__,
    )
    for worker_id in range(data_loader_workers):
        proc = _MP_CTX.Process(
            target=pbg_batch_worker,
            args=(
                source,
                worker_id,
                data_loader_workers,
                global_worker_offset,
                global_num_workers,
                seed,
                batch_size,
                config,
                vocab,
                augmenter,
                batch_queue,
                shard_pass,
            ),
            daemon=True,
        )
        proc.start()
        procs.append(proc)
    log.info(
        "spawned %d data-loader processes in %.1fs",
        data_loader_workers,
        time.monotonic() - t_spawn,
    )
    try:
        first_batch = True
        last_heartbeat = t_spawn
        while True:
            try:
                item = batch_queue.get(timeout=60.0)
            except queue.Empty:
                item = None
            if item is None:
                if first_batch:
                    waited = time.monotonic() - t_spawn
                    _loader_heartbeat(
                        "still waiting for first batch after %.0fs "
                        "(workers alive: %d/%d, deadline %.0fs)",
                        waited,
                        sum(1 for proc in procs if proc.is_alive()),
                        len(procs),
                        _FIRST_BATCH_DEADLINE_S,
                    )
                    if waited > _FIRST_BATCH_DEADLINE_S:
                        alive = [proc.pid for proc in procs if proc.is_alive()]
                        raise TimeoutError(
                            "no training batch after "
                            f"{waited:.0f}s; {len(alive)}/{len(procs)} loader "
                            f"workers alive (pids={alive}). Suspect per-frame "
                            "tokenization stall or wedged NFS reads; inspect the "
                            "assigned shard of a live worker."
                        )
                elif time.monotonic() - last_heartbeat > 600.0:
                    last_heartbeat = time.monotonic()
                    _loader_heartbeat(
                        "no batch for 600s after first batch "
                        "(workers alive: %d/%d)",
                        sum(1 for proc in procs if proc.is_alive()),
                        len(procs),
                    )
                continue
            if isinstance(item, BaseException):
                raise item
            if first_batch:
                _loader_heartbeat(
                    "first collated batch from data-loader workers after %.1fs",
                    time.monotonic() - t_spawn,
                )
                first_batch = False
            yield item
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=1.0)


# ── Joint single-pass loader ───────────────────────────────────────


def molecular_batches(model, source, *, batch_size, seed, start_step=0, audit=None, cursor_tracker=None):
    """Yield the batches of one deterministic pass over a joint molecular dataset.

    Resume replays CPU preparation and skips already trained batches. It never
    repeats a frame in the restored model trajectory.
    """
    config, vocab = model.config, model.vocab
    root = Path(source['dir'])
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if manifest.get('release_contract') not in {'many_molecules_pilot_v1', 'many_molecules_joint_v1'}:
        raise ValueError('finite molecular loader requires an approved molecular manifest')
    tokeniser = make_tokeniser(vocab, prompt_format=config.prompt_format)
    augmenter = augmenter_from_config(config)
    append_only = bool(source.get('molecular_append_only'))
    contract = dict(seed=seed, batch_size=batch_size, model_config=asdict(config))
    # Block recomputation changes memory use only, not data preparation; it is
    # left out so that cursors saved before the field existed stay valid.
    contract['model_config'].pop('remat_blocks')
    if config.atom_order == 'source_v1':
        # Keep pre-existing source-order cursors byte-for-byte compatible.
        contract['model_config'].pop('atom_order')
        contract['model_config'].pop('atom_order_implementation')
    else:
        contract['atom_order_implementation'] = config.atom_order_implementation
    shard_ids = [dict(path=Path(s['path']).name, sha256=s.get('sha256')) for s in manifest['shards']]
    saved = cursor_tracker.molecular if cursor_tracker is not None else None
    start_shard = start_frame = prior_consumed = 0
    if append_only and start_step:
        if not saved or saved.get('step') != start_step:
            raise ValueError('append-only resume requires its exact trained-batch molecular cursor')
        # Cursors saved before this release also hold the retired model options.
        saved_contract = dict(saved['contract'], model_config=retire('model', saved['contract']['model_config']))
        if saved_contract != contract:
            raise ValueError('molecular resume changed batch size, seed, or model preparation')
        old_shards = saved['shards']
        if shard_ids[:len(old_shards)] != old_shards:
            raise ValueError('molecular dataset expansion must retain the exact old shard prefix')
        if saved.get('closed') and len(shard_ids) != len(old_shards):
            raise ValueError('cannot append after training the padded final batch; resume an earlier stable checkpoint')
        start_shard, start_frame = saved['next_shard'], saved['next_frame']
        prior_consumed = saved['trained_frames']
    rows, row = [], []
    remaining = config.max_seq_len
    batch_index = start_step if append_only else 0
    consumed = prior_consumed
    for shard_index, shard in enumerate(manifest['shards']):
        if shard_index < start_shard:
            continue
        path = root / Path(shard['path']).name
        for frame_index, original in enumerate(iter_frames_from_tar(path, adapter='n4_smiles',
            sample_layout='n4_json_pair_v1')):
            if shard_index == start_shard and frame_index < start_frame:
                continue
            # FreeSolv contains water itself as a solute. In vacuum/implicit
            # records it must not be classified as an explicit solvent block.
            if original.environment != 'explicit_water' and original.metadata['n_waters']:
                original = replace(original, metadata=dict(original.metadata,
                    n_waters=0, n_solute_atoms=len(original.conformer)))
            frame, rng = select_representation(original, seed)
            fitted = fit_frame(frame, config, tokeniser, remaining)
            if fitted is None and row:
                rows.append(row)
                row, remaining = [], config.max_seq_len
                fitted = fit_frame(frame, config, tokeniser, remaining)
            if fitted is None:
                # One deterministic fallback, never a second data record.
                frame = replace(frame, helm=original.helm,
                    metadata=dict(frame.metadata, helm_rewrite_fallback=True))
                fitted = fit_frame(frame, config, tokeniser, config.max_seq_len)
            if fitted is None:
                raise ValueError(f'solute does not fit: {original.metadata["pilot_key"]}')
            prepared = prepare_frame(fitted, config, vocab, augmenter,
                rng=rng, tokeniser=tokeniser)
            if prepared is None or prepared.seg_len > remaining:
                raise ValueError('molecular preparation unexpectedly rejected a source frame')
            row.append(prepared)
            remaining -= prepared.seg_len
            consumed += 1
            if audit is not None:
                audit(original, fitted, prepared)
            if len(rows) == batch_size:
                if batch_index >= start_step:
                    batch = collate_tokenised_frames(config, vocab, rows)
                    if append_only:
                        # The current frame is the first row of the NEXT batch.
                        # Prefetch can advance it, but only note_trained_batch commits this cursor.
                        batch[BATCH_MOLECULAR_KEY] = dict(step=batch_index + 1,
                            next_shard=shard_index, next_frame=frame_index,
                            trained_frames=consumed - 1, contract=contract, shards=shard_ids, closed=False)
                    yield batch
                batch_index += 1
                rows = []
    if row:
        rows.append(row)
    if rows:
        # Empty padded rows contain no valid samples or loss tokens.
        rows.extend([] for _ in range(batch_size - len(rows)))
        if batch_index >= start_step:
            batch = collate_tokenised_frames(config, vocab, rows)
            if append_only:
                batch[BATCH_MOLECULAR_KEY] = dict(step=batch_index + 1,
                    next_shard=len(shard_ids), next_frame=0, trained_frames=consumed,
                    contract=contract, shards=shard_ids, closed=True)
            yield batch
        batch_index += 1
    if consumed != manifest['total_samples']:
        raise ValueError('finite pass sample count disagrees with manifest')
    if start_step > batch_index:
        raise ValueError('resume step exceeds finite dataset')


# ── Loader worker processes ────────────────────────────────────────


_CPU_ONLY_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_PLATFORM_NAME": "cpu",
    "ROCR_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "CUDA_VISIBLE_DEVICES": "",
}


def hide_accelerator_devices() -> None:
    for key, value in _CPU_ONLY_ENV.items():
        os.environ[key] = value


def pbg_batch_worker(
    source: dict[str, Any],
    worker_id: int,
    num_workers: int,
    global_worker_offset: int | None,
    global_num_workers: int | None,
    seed: int,
    batch_size: int,
    config: PBGConfig,
    vocab: dict[str, object],
    augmenter: Augmenter,
    batch_queue: mp.Queue[Any],
    shard_pass: tuple[int, ...] | None = None,
) -> None:
    """Entry point of a spawned data-loader process: queue its collated batches."""
    hide_accelerator_devices()
    try:
        for batch in _iter_tagged_collated_batches(
            source,
            worker_id=worker_id,
            num_workers=num_workers,
            global_worker_offset=global_worker_offset,
            global_num_workers=global_num_workers,
            seed=seed,
            batch_size=batch_size,
            config=config,
            vocab=vocab,
            augmenter=augmenter,
            shard_pass=shard_pass,
        ):
            batch_queue.put(batch)
    except BaseException as exc:
        log.exception("decode worker %d failed", worker_id)
        batch_queue.put(exc)
