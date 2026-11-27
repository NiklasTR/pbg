"""Rollout export: energies, stereo repair, importance weights and ensemble files."""

from __future__ import annotations

import gc
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pbg.core.constants import GAS_CONSTANT_KJ_MOL_K
from pbg.core.ensemble import Ensemble
from pbg.core.frame import DataFrame
from pbg.core.helm import mapped_smiles_to_chem_helm
from pbg.core.molecule import atom_numbers_from_smiles
from pbg.core.peptide import sequence_to_smiles
from pbg.core.stereo import (
    compile_stereo_spec,
    configured_stereo_key,
    key_to_mapped_smiles,
    stereo_groups,
    stereo_keys,
)
from pbg.energy import ProtocolEvaluator
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.model import PBG
from pbg.models.pbg.sample import SampleBatch, sample_molecular_conformers
from pbg.models.pbg.system import MolecularSystem

EnergyScorer = Callable[[MolecularSystem, np.ndarray], np.ndarray]


def energy_frames(
    frames: list[DataFrame] | tuple[DataFrame, ...],
    *,
    sequence: str,
    environment: str,
    platform_name: str | None = None,
) -> np.ndarray:
    """Score peptide frames (kJ/mol) with the ff14SB protocol of their environment.

    Atoms beyond the peptide topology are OHH waters (ff14sb-tip3p-rf2nm).
    Without waters, ``vacuum`` frames use ff14sb and all others ff14sb-obc1-2nm.
    """
    if not frames or any(frame.conformer is None for frame in frames):
        raise ValueError("frames must be non-empty and all have conformers")
    samples = np.asarray([frame.conformer for frame in frames], dtype=np.float64)
    peptide_n = int(atom_numbers_from_smiles(sequence_to_smiles(sequence)).shape[0])
    extra = samples.shape[1] - peptide_n
    if extra < 0 or extra % 3 != 0:
        raise ValueError(f"cannot infer n_waters: peptide has {peptide_n} atoms, frame has {samples.shape[1]}")
    target = "ff14sb-tip3p-rf2nm" if extra else "ff14sb" if environment == "vacuum" else "ff14sb-obc1-2nm"
    return ProtocolEvaluator.peptide(sequence, target, n_waters=extra // 3, platform_name=platform_name)(samples)


@dataclass(frozen=True, slots=True)
class BatchSizeTrial:
    batch_size: int
    seconds: float
    samples_per_second: float


def require_gpu_backend() -> None:
    """Fail before sampling if JAX silently fell back to the CPU."""
    backend = jax.default_backend()
    if backend != "gpu":
        raise RuntimeError(
            "PBG sampling requires a GPU JAX backend; "
            f"got {backend!r} with devices={jax.devices()}"
        )


def tune_sampling_batch_size(
    model: PBG,
    system: MolecularSystem,
    key: Array,
    *,
    candidates: tuple[int, ...] = (16, 32, 64, 128, 256, 512),
    proposal_temperature: float = 1.0,
) -> tuple[int, tuple[BatchSizeTrial, ...]]:
    """Benchmark complete KV-cached decodes and return the fastest batch size."""
    require_gpu_backend()
    candidates = tuple(sorted({int(value) for value in candidates if int(value) > 0}))
    if not candidates:
        raise ValueError("at least one positive batch-size candidate is required")
    trials: list[BatchSizeTrial] = []
    for batch_size in candidates:
        try:
            key, warmup_key, timed_key = jax.random.split(key, 3)
            sample_molecular_conformers(
                model,
                system,
                warmup_key,
                num_samples=batch_size,
                batch_size=batch_size,
                proposal_temperature=proposal_temperature,
            )
            started = time.perf_counter()
            sample_molecular_conformers(
                model,
                system,
                timed_key,
                num_samples=batch_size,
                batch_size=batch_size,
                proposal_temperature=proposal_temperature,
            )
            elapsed = time.perf_counter() - started
        except Exception as error:
            message = str(error).lower()
            recoverable = (
                "resource_exhausted" in message
                or "out of memory" in message
                or "autotuning failed" in message
                or "failed to get configs" in message
            )
            if not recoverable:
                raise
            continue
        finally:
            jax.clear_caches()
            gc.collect()
        trials.append(
            BatchSizeTrial(
                batch_size=batch_size,
                seconds=elapsed,
                samples_per_second=batch_size / elapsed,
            )
        )
    if not trials:
        raise RuntimeError(
            "no sampling batch-size candidate compiled and fit in device memory"
        )
    best = max(trials, key=lambda trial: trial.samples_per_second)
    return best.batch_size, tuple(trials)


def repair_all_mirrored_stereo(
    samples: np.ndarray,
    mapped_smiles: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror fully resolved all-inverted stereoisomers into the intended state.

    Returns repaired samples, the repaired-frame mask, and a mask covering both
    intended and all-mirrored frames. The masks describe geometry only; policy
    likelihood always belongs to the original generated conformer.
    """
    conformers = np.asarray(samples, dtype=np.float64)
    spec = compile_stereo_spec(mapped_smiles)
    intended = configured_stereo_key(mapped_smiles, spec)
    if intended is None or spec.atoms.size == 0:
        return (
            conformers.copy(),
            np.zeros(conformers.shape[0], dtype=bool),
            np.zeros(conformers.shape[0], dtype=bool),
        )
    config, resolved = stereo_keys(spec, jnp.asarray(conformers[:, :, 1:4]))
    config = np.asarray(jax.device_get(config), dtype=np.uint32)
    resolved = np.asarray(jax.device_get(resolved), dtype=np.uint32)
    full = np.uint32((1 << int(spec.atoms.shape[0])) - 1)
    intended_key = np.uint32(intended)
    mirrored_key = intended_key ^ full
    fully_resolved = resolved == full
    repair = fully_resolved & (config == mirrored_key)
    fold = fully_resolved & ((config == intended_key) | (config == mirrored_key))
    repaired = conformers.copy()
    repaired[repair, :, 1:4] *= -1.0
    return repaired, repair, fold


def ensemble_filename(
    system: MolecularSystem,
    *,
    config_key: int | None = None,
    resolved_key: int | None = None,
) -> str:
    def slug(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "molecule"

    temperature = f"{system.temperature_K:g}K"
    identity = (
        "intended"
        if config_key is None
        else f"stereo-r{int(resolved_key):08x}-c{int(config_key):08x}"
    )
    return f"{slug(system.id)}__{slug(system.environment)}__{temperature}__{identity}.npz"


def build_sampled_ensembles(
    system: MolecularSystem,
    batch: SampleBatch,
    energy_kj_mol: np.ndarray,
    *,
    config: PBGConfig,
    generator_checkpoint: str | None = None,
    stereo_repair_mask: np.ndarray | None = None,
    reflection_folding_enabled: bool = False,
) -> tuple[tuple[str, Ensemble], ...]:
    energy = np.asarray(energy_kj_mol, dtype=np.float64)
    if energy.shape != (batch.samples.shape[0],):
        raise ValueError("energy must contain one value per sampled frame")
    spec = compile_stereo_spec(system.mapped_smiles)
    config_key, resolved_key = stereo_keys(spec, jnp.asarray(batch.samples[:, :, 1:4]))
    config_key = np.asarray(jax.device_get(config_key), dtype=np.uint32)
    resolved_key = np.asarray(jax.device_get(resolved_key), dtype=np.uint32)
    unique_config, unique_resolved, inverse, counts = stereo_groups(
        config_key, resolved_key
    )
    full_key = (1 << int(spec.atoms.shape[0])) - 1
    intended_key = configured_stereo_key(system.mapped_smiles, spec)
    minimum_count = int(config.stereo_minimum_count)
    promoted = [
        index
        for index, (resolved, count) in enumerate(
            zip(unique_resolved, counts, strict=True)
        )
        if int(resolved) == full_key and int(count) >= minimum_count
    ]
    output_identities: list[tuple[int | None, int | None, int | None, bool]] = [
        (intended_key, full_key if intended_key is not None else None, None, True)
    ]
    output_identities.extend(
        (int(unique_config[index]), int(unique_resolved[index]), index, False)
        for index in promoted
        if intended_key is None or int(unique_config[index]) != intended_key
    )
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * system.temperature_K)
    log_weight = -beta * energy - batch.logq
    base_valid = np.isfinite(energy) & np.isfinite(batch.logq)
    outputs = []
    repair_mask = (
        np.zeros(batch.samples.shape[0], dtype=bool)
        if stereo_repair_mask is None
        else np.asarray(stereo_repair_mask, dtype=bool)
    )
    if repair_mask.shape != (batch.samples.shape[0],):
        raise ValueError("stereo_repair_mask must contain one value per sampled frame")
    for output_config, output_resolved, group_index, is_intended in output_identities:
        if group_index is None and output_config is not None:
            eligible = (
                base_valid & (resolved_key == full_key) & (config_key == output_config)
            )
        elif group_index is None:
            eligible = base_valid & (resolved_key == full_key)
        else:
            eligible = base_valid & (inverse == group_index)
        if is_intended:
            output_indices = np.arange(batch.samples.shape[0], dtype=np.int64)
        else:
            output_indices = np.flatnonzero(inverse == group_index)
        output_eligible = eligible[output_indices]
        weights = np.zeros(output_indices.size, dtype=np.float64)
        frame_valid = np.zeros(output_indices.size, dtype=np.bool_)
        alpha = None
        if int(np.sum(eligible)) >= minimum_count:
            # Self-normalised importance weights of the whole pool, mean one;
            # rollouts store them under the historical name
            # ``self_distillation_weight`` (the paper's ``naive`` rule).
            pool_indices = np.flatnonzero(eligible)
            pool_log_weight = log_weight[pool_indices]
            if not np.all(np.isfinite(pool_log_weight)):
                raise ValueError("log_weight must be finite")
            relative = np.exp(np.clip(pool_log_weight - float(np.max(pool_log_weight)), -700.0, 0.0))
            total = float(np.sum(relative))
            if total <= 0.0:
                raise RuntimeError("importance weights have zero mass")
            local_retained = np.searchsorted(output_indices, pool_indices)
            weights[local_retained] = (relative / total) * float(relative.size)
            frame_valid[local_retained] = True
            alpha = 1.0
        mapped_smiles = system.mapped_smiles
        helm = system.helm
        if output_config is not None and output_resolved is not None:
            rendered = key_to_mapped_smiles(
                system.mapped_smiles, spec, output_config, output_resolved
            )
            if rendered is not None:
                mapped_smiles = rendered
                if output_config != intended_key:
                    helm = mapped_smiles_to_chem_helm(rendered)
        name = ensemble_filename(
            system,
            config_key=None if group_index is None else output_config,
            resolved_key=output_resolved,
        )
        ensemble = Ensemble(
            samples=batch.samples[output_indices],
            smiles=mapped_smiles,
            id=Path(name).stem,
            sequence=system.sequence,
            helm=helm,
            environment=system.environment,
            temperature=system.temperature_K,
            energy=energy[output_indices],
            logq=batch.logq[output_indices],
            self_distillation_weight=weights,
            behavior_token_log_probs=None
            if batch.behavior_token_log_probs is None
            else batch.behavior_token_log_probs[output_indices],
            behavior_token_ids=None
            if batch.behavior_token_ids is None
            else batch.behavior_token_ids[output_indices],
            frame_valid=frame_valid,
            stereo_valid=output_eligible,
            source="pbg_sample",
            generator_checkpoint=generator_checkpoint,
            alpha=alpha,
            ensemble_metadata={
                "minimum_stereo_count": minimum_count,
                # The paper rollouts store the plain ("naive") importance weights.
                "self_distillation_condition": "naive",
                "fit_alpha": False,
                "inclusion_nats": None,
                "target_ress": None,
                "stereo_config_key": -1 if output_config is None else output_config,
                "stereo_resolved_key": -1
                if output_resolved is None
                else output_resolved,
                "stereo_pool_count": int(np.sum(eligible)),
                "output_frame_count": int(output_indices.size),
                "valid_output_frame_count": int(np.sum(output_eligible)),
                "all_mirrored_stereo_repair": bool(np.any(repair_mask)),
                "all_mirrored_stereo_repair_count": int(
                    np.sum(repair_mask[output_indices])
                ),
                "reconstruction": "center",
                "enantiomer_correction": "repaired_total_reflection_v1"
                if reflection_folding_enabled
                else "none",
                "proposal_density": "generated",
                "logq_source": "original_generated_conformer",
                "reflection_density_recomputed": False,
                "validity_filter": "midpoint_enantiocorrected_intended_stereo"
                if reflection_folding_enabled
                else "recomputed_finite_intended_stereochemistry",
            },
        )
        outputs.append((name, ensemble))
    return tuple(outputs)


def sample_ensembles(
    model: PBG,
    system: MolecularSystem,
    key: Array,
    *,
    energy_scorer: EnergyScorer,
    num_samples: int,
    output_dir: str | Path,
    batch_size: int = 64,
    proposal_temperature: float = 1.0,
    generator_checkpoint: str | None = None,
    repair_all_mirrored: bool = False,
    model_generation: str = 'auto',
) -> tuple[Path, ...]:
    batch = sample_molecular_conformers(
        model,
        system,
        key,
        num_samples=num_samples,
        batch_size=batch_size,
        proposal_temperature=proposal_temperature,
        model_generation=model_generation,
    )
    return export_sample_batch(
        model,
        system,
        batch,
        energy_scorer=energy_scorer,
        output_dir=output_dir,
        generator_checkpoint=generator_checkpoint,
        repair_all_mirrored=repair_all_mirrored,
    )


def export_sample_batch(
    model: PBG,
    system: MolecularSystem,
    batch: SampleBatch,
    *,
    energy_scorer: EnergyScorer,
    output_dir: str | Path,
    generator_checkpoint: str | None = None,
    repair_all_mirrored: bool = False,
) -> tuple[Path, ...]:
    """Score and export an existing exact-target molecular sample batch.

    With ``repair_all_mirrored``, all-mirrored frames are reflected into the
    intended stereoisomer. Correction never calls the model on reflected
    coordinates: each frame keeps the policy log-likelihood it was generated with.
    """
    stereo_repair_mask = None
    if repair_all_mirrored:
        repaired, stereo_repair_mask, _ = repair_all_mirrored_stereo(batch.samples, system.mapped_smiles)
        if np.any(stereo_repair_mask):
            # Generated token traces describe the pre-repair coordinates and
            # must not be reused for behavior-policy training after folding.
            batch = SampleBatch(samples=repaired, logq=batch.logq,
                                behavior_token_log_probs=None, behavior_token_ids=None)
    energies = energy_scorer(system, batch.samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, ensemble in build_sampled_ensembles(
        system,
        batch,
        energies,
        config=model.config,
        generator_checkpoint=generator_checkpoint,
        stereo_repair_mask=stereo_repair_mask,
        reflection_folding_enabled=repair_all_mirrored,
    ):
        path = output_dir / name
        ensemble.to_npz(path)
        paths.append(path)
    return tuple(paths)
